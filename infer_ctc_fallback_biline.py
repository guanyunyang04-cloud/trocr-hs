# -*- coding: utf-8 -*-
# infer_ctc_fallback_biline.py
import os, sys, math, time, glob, json, argparse
from typing import List, Tuple, Dict, Any
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# -------------------- utils --------------------
def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
        return torch.device("npu")
    return torch.device("cpu")

def cer(pred: str, ref: str) -> float:
    m, n = len(ref), len(pred)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0]=i
    for j in range(n+1): dp[0][j]=j
    for i in range(1,m+1):
        ri = ref[i-1]
        for j in range(1,n+1):
            dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(0 if ri==pred[j-1] else 1))
    return dp[m][n]/max(1,m)

# -------------------- vocab --------------------
class CharVocab:
    def __init__(self, chars: str, blank_id: int = 0):
        uniq=[]
        for c in chars:
            if c not in uniq: uniq.append(c)
        self.chars=["<blank>"]+uniq
        self.blank_id=blank_id
        self.stoi={c:i for i,c in enumerate(self.chars)}
        self.itos={i:c for c,i in self.stoi.items()}

    @staticmethod
    def load(model_dir: str, fallback_chars: str=None):
        f=os.path.join(model_dir,"ctc_vocab.json")
        if os.path.exists(f):
            j=json.load(open(f,"r",encoding="utf-8"))
            chars=j.get("chars")
            if isinstance(chars,list):
                chars="".join([c for c in chars if c!="<blank>"])
            blank=j.get("blank_id",0)
            return CharVocab(chars,blank)
        if fallback_chars:
            print("[WARN] no ctc_vocab.json, use --ctc_chars")
            return CharVocab(fallback_chars,0)
        raise FileNotFoundError("ctc_vocab.json not found and --ctc_chars is empty")

    def encode(self, s:str)->List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]
    def decode(self, ids:List[int])->str:
        return "".join(self.itos[i] for i in ids if i!=self.blank_id)

# -------------------- 双头行感知 CTCHead --------------------
class CTCHead(nn.Module):
    """
    2D门控 -> top/bot 两条序列 -> 共享1D头 -> 分类
    forward(enc_tokens, Hp, Wp, return_gates=False)
      返回 ((logp_top, logits_top), (logp_bot, logits_bot)) 以及可选 (gate_top_mean, gate_bot_mean)
    """
    def __init__(self, d_in:int, vocab_size:int, blank_id:int=0):
        super().__init__()
        mid=max(32, d_in//8)
        self.g1=nn.Conv2d(d_in, mid, 3, padding=1)
        self.g2=nn.Conv2d(mid, 2, 1)
        self.pw =nn.Linear(d_in, d_in, bias=False)
        self.dw1=nn.Conv1d(d_in, d_in, 3, padding=1, groups=d_in)
        self.dw2=nn.Conv1d(d_in, d_in, 3, padding=1, groups=d_in)
        self.cls=nn.Linear(d_in, vocab_size)
        self.blank_id=blank_id

    def _pool_line(self, x_hw_d:torch.Tensor, w_hw:torch.Tensor):
        # x: (B,H',W',D)  w: (B,H',W')
        denom = w_hw.unsqueeze(-1).sum(1).clamp_min(1e-6)  # (B,W',1)
        x = (x_hw_d * w_hw.unsqueeze(-1)).sum(1) / denom   # (B,W',D)
        res=x
        x=self.pw(x)
        x=x.transpose(1,2)
        x=F.gelu(self.dw1(x)); x=F.gelu(self.dw2(x))
        x=x.transpose(1,2)+res
        logits=self.cls(x)                                  # (B,W',C)
        logp=logits.log_softmax(-1).transpose(0,1)          # (T=W',B,C)
        return logp, logits

    def forward(self, enc_tokens:torch.Tensor, Hp:int, Wp:int, return_gates:bool=False):
        w = next(self.parameters())
        if enc_tokens.device!=w.device or enc_tokens.dtype!=w.dtype:
            enc_tokens = enc_tokens.to(device=w.device, dtype=w.dtype)

        B,N,D = enc_tokens.shape
        target=Hp*Wp
        extra=N-target
        if extra in (1,2): enc_tokens=enc_tokens[:,extra:,:]; N-=extra
        assert N==target, f"CTCHead: N={N} != H'W'={target}"

        x = enc_tokens.view(B,Hp,Wp,D)                    # (B,H',W',D)
        w2= self.g2(F.gelu(self.g1(x.permute(0,3,1,2))))  # (B,2,H',W')
        w2= F.softmax(w2,dim=1)
        w_top = w2[:,0].permute(0,1,2)                    # (B,H',W')
        w_bot = w2[:,1].permute(0,1,2)

        logp_t, logits_t = self._pool_line(x, w_top)
        logp_b, logits_b = self._pool_line(x, w_bot)
        if return_gates:
            g_top = w_top.mean(dim=(1,2))
            g_bot = w_bot.mean(dim=(1,2))
            return (logp_t,logits_t),(logp_b,logits_b),(g_top,g_bot)
        return (logp_t,logits_t),(logp_b,logits_b)

# -------------------- CTC 解码/打分/门控 --------------------
def np_topk_idx(x:np.ndarray, k:int)->List[int]:
    if k>=x.size: return list(np.argsort(-x))
    idx = np.argpartition(-x, k-1)[:k]
    return list(idx[np.argsort(-x[idx])])

def ctc_beam_search(log_probs:torch.Tensor, beam=10, blank_id=0, topk_each_step=12):
    """
    纯 numpy 的 beam（避免 GPU/CPU dtype 冲突）
    log_probs: (T,1,C)  → 返回 (ids, best_logp, avg_margin)
    """
    T,B,C = log_probs.shape
    assert B==1
    beams = {(): (0.0, -np.inf)}  # path -> (p_blank, p_nonblank)
    for t in range(T):
        lp = log_probs[t,0].detach().float().cpu().numpy()  # (C,)
        nxt = defaultdict(lambda: (-np.inf, -np.inf))
        # blank
        for pref,(pb,pnb) in beams.items():
            nb = nxt[pref]
            nb = (np.logaddexp.reduce([nb[0], pb+lp[blank_id], pnb+lp[blank_id]]), nb[1])
            nxt[pref]=nb
        # symbols
        for c in np_topk_idx(lp, min(topk_each_step,C)):
            if c==blank_id: continue
            for pref,(pb,pnb) in beams.items():
                if len(pref)>0 and c==pref[-1]:
                    nb = nxt[pref]
                    nb = (nb[0], np.logaddexp(nb[1], pb+lp[c]))
                    nxt[pref]=nb
                else:
                    newp = pref+(c,)
                    nb = nxt[newp]
                    nb = (nb[0], np.logaddexp.reduce([nb[1], pb+lp[c], pnb+lp[c]]))
                    nxt[newp]=nb
        # prune
        scored = [(p, np.logaddexp(pb,pnb)) for p,(pb,pnb) in nxt.items()]
        scored.sort(key=lambda z: z[1], reverse=True)
        beams = dict(scored[:beam])
    if not beams: return [], -1e9, 0.0
    best = sorted([(list(p),s) for p,s in beams.items()], key=lambda z:z[1], reverse=True)
    (seq1,s1) = best[0]
    (seq2,s2) = best[1] if len(best)>1 else (seq1, s1-1e3)
    margin = (s1-s2)/max(1,len(seq1))
    return seq1, float(s1), float(margin)

def ctc_confidence(logits:torch.Tensor, blank_id=0):
    # logits: (B,W',C)
    probs = logits.softmax(-1)
    maxp, maxi = probs.max(-1)
    blank = (maxi==blank_id).float()
    conf_avg = maxp.mean(dim=1)            # (B,)
    peak_ratio = 1.0 - blank.mean(dim=1)   # (B,)
    entropy = -(probs*(probs.clamp_min(1e-9).log())).sum(-1).mean(1)
    return conf_avg, peak_ratio, entropy

def need_fallback(conf, margin, peak_ratio, thr_conf=0.90, thr_margin=0.30, pr_lo=0.35, pr_hi=0.85):
    return (conf < thr_conf) or (margin < thr_margin) or not (pr_lo <= peak_ratio <= pr_hi)

# ----- 行合并 & 重复行抑制 -----
def _cer_str(a: str, b: str) -> float:
    m, n = len(b), len(a)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0]=i
    for j in range(n+1): dp[0][j]=j
    for i in range(1,m+1):
        rb=b[i-1]
        for j in range(1,n+1):
            dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1] + (0 if rb==a[j-1] else 1))
    return dp[m][n]/max(1,m)

def merge_two_lines(txt_t, txt_b,
                    conf_t, conf_b, pr_t, pr_b,
                    gate_t, gate_b,
                    min_len=2, pr_min=0.25, conf_min=0.65,
                    dup_sim_thr=0.6, gate_dom_thr=0.78):
    # 1) 门控占优：单行
    if max(gate_t,gate_b)>=gate_dom_thr and min(gate_t,gate_b)<=1.0-gate_dom_thr:
        return (txt_t,"top") if gate_t>gate_b else (txt_b,"bot")
    # 2) 过滤弱行
    keep_t = (len(txt_t)>=min_len) and (pr_t>=pr_min) and (conf_t>=conf_min)
    keep_b = (len(txt_b)>=min_len) and (pr_b>=pr_min) and (conf_b>=conf_min)
    # 3) 两行都似乎有效但高度相似（重复）→ 选强的一行
    if keep_t and keep_b:
        sim = 1.0 - _cer_str(txt_t, txt_b)  # 0~1
        if sim >= dup_sim_thr:
            score_t = conf_t + 0.5*pr_t
            score_b = conf_b + 0.5*pr_b
            return (txt_t,"top") if score_t>=score_b else (txt_b,"bot")
    # 4) 正常拼接/单行
    if keep_t and keep_b: return txt_t+" "+txt_b, "2line"
    if keep_t: return txt_t, "top"
    if keep_b: return txt_b, "bot"
    # 5) 两行都弱：保留分更高的一行
    score_t = conf_t + 0.5*pr_t
    score_b = conf_b + 0.5*pr_b
    return (txt_t,"top?") if score_t>=score_b else (txt_b,"bot?")

# -------------------- dataset --------------------
class ImgCSV(Dataset):
    def __init__(self, csv_path, image_root, processor,
                 image_col=None, label_col=None, basename_lookup=False,
                 exts=".jpg,.jpeg,.png,.bmp"):
        self.df=pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        self.df.columns=[c.strip() for c in self.df.columns]
        raw=list(self.df.columns)
        def norm(s): return s.strip().lower().replace(" ","").replace("_","").replace("-","")
        n2r={}
        for c in raw:
            k=norm(c)
            if k not in n2r: n2r[k]=c
        def resolve(col, cands, what):
            if col:
                if col in raw: return col
                nk=norm(col)
                if nk in n2r: return n2r[nk]
                raise ValueError(f"--{what}='{col}' not found; csv cols={raw}")
            for cand in cands:
                if cand in raw: return cand
                nk=norm(cand)
                if nk in n2r: return n2r[nk]
            return None
        img_cands=["imagepath","filepath","path","filename","file","img","imagename","imgpath","imgfile","file_name","image_name"]
        lab_cands=["label","text","gt","target","transcript","labeltext","label_text"]
        self.ic=resolve(image_col,img_cands,"image_col")
        self.lc=resolve(label_col, lab_cands,"label_col")
        if self.ic is None: raise ValueError(f"no image_col found; csv cols={raw}")
        print(f"[INFO] CSV columns={raw}; using image_col='{self.ic}' label_col='{self.lc}'")
        self.root=image_root; self.proc=processor
        self.basename_lookup=basename_lookup
        self.exts=tuple([e.strip().lower() for e in exts.split(",") if e.strip()])
        self.name2path=None
        if basename_lookup:
            self.name2path={}
            for dp,_,files in os.walk(self.root):
                for f in files:
                    fl=f.lower()
                    if fl.endswith(self.exts):
                        self.name2path.setdefault(fl, os.path.join(dp,f))
    def __len__(self): return len(self.df)
    def _resolve(self, val):
        s=str(val).strip()
        if os.path.isabs(s): return s
        if "/" in s or "\\" in s: return os.path.join(self.root,s)
        if self.basename_lookup and self.name2path is not None:
            p=self.name2path.get(s.lower())
            if p is None: raise FileNotFoundError(f"basename {s} not found in {self.root}")
            return p
        return os.path.join(self.root,s)
    def __getitem__(self, idx):
        r=self.df.iloc[idx]
        p=self._resolve(r[self.ic])
        lab=str(r[self.lc]) if self.lc is not None else None
        im=Image.open(p).convert("RGB")
        pv=self.proc(images=im, return_tensors="pt").pixel_values[0]
        return {"pixel_values":pv, "path":p, "label":lab}

class ImgDir(Dataset):
    def __init__(self, images_dir, processor, exts=(".jpg",".jpeg",".png",".bmp")):
        self.paths=[p for e in exts for p in glob.glob(os.path.join(images_dir,f"**/*{e}"),recursive=True)]
        if not self.paths: raise FileNotFoundError(f"No images found under {images_dir}")
        self.proc=processor
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p=self.paths[idx]
        im=Image.open(p).convert("RGB")
        pv=self.proc(images=im, return_tensors="pt").pixel_values[0]
        return {"pixel_values":pv, "path":p, "label":None}

# -------------------- grid helper --------------------
def guess_grid_from_N(N:int)->Tuple[int,int,int]:
    s=int(round(N**0.5))
    for d in (0,1,2):
        if s*s+d==N: return s,s,d
    s=int(max(1, math.sqrt(max(1,N))))
    return s,s,max(0,N-s*s)

# -------------------- main --------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--images_dir", type=str, default=None)
    ap.add_argument("--images_csv", type=str, default=None)
    ap.add_argument("--image_root", type=str, default=".")
    ap.add_argument("--image_col", type=str, default=None)
    ap.add_argument("--label_col", type=str, default=None)
    ap.add_argument("--basename_lookup", action="store_true", default=False)

    ap.add_argument("--ctc_chars", type=str, default=None,
        help="若无 ctc_vocab.json，用此字符集（务必含空格）。示例：'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '")
    ap.add_argument("--dup_sep", type=str, default="|",
        help="若训练时使用重复分隔符（如 |），推理解码会删除它；否则忽略")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--beam", type=int, default=12)

    ap.add_argument("--conf_thr", type=float, default=0.80)
    ap.add_argument("--margin_thr", type=float, default=0.15)
    ap.add_argument("--pr_lo", type=float, default=0.20)
    ap.add_argument("--pr_hi", type=float, default=0.95)

    ap.add_argument("--gen_max_new_tokens", type=int, default=24)
    ap.add_argument("--gen_beams", type=int, default=1)

    ap.add_argument("--out_csv", type=str, required=True)
    args=ap.parse_args()

    dev=pick_device()
    print(f"[INFO] device={dev}")

    processor=TrOCRProcessor.from_pretrained(args.model_dir)
    model=VisionEncoderDecoderModel.from_pretrained(args.model_dir).eval().to(dev)
    vocab=CharVocab.load(args.model_dir, fallback_chars=args.ctc_chars)

    # 安装双头 CTCHead 并加载权重
    dpar=next(model.parameters()); dtype=dpar.dtype
    if not hasattr(model,"ctc_head"):
        model.ctc_head=CTCHead(model.config.encoder.hidden_size, len(vocab.chars), blank_id=vocab.blank_id).to(device=dev, dtype=dtype)
    else:
        model.ctc_head.to(device=dev, dtype=dtype)

    # 从 bin 提取 ctc_head.* 权重
    binp=os.path.join(args.model_dir,"pytorch_model.bin")
    if os.path.exists(binp):
        sd=torch.load(binp, map_location="cpu")
        sub={k.replace("ctc_head.",""):v for k,v in sd.items() if k.startswith("ctc_head.")}
        if sub:
            miss, unexp = model.ctc_head.load_state_dict(sub, strict=False)
            print(f"[INFO] loaded CTC head weights (missing={len(miss)}, unexpected={len(unexp)})")
    model.ctc_head.to(device=dev, dtype=dtype)

    # 数据
    if args.images_dir:
        dataset=ImgDir(args.images_dir, processor)
    elif args.images_csv:
        dataset=ImgCSV(args.images_csv, args.image_root, processor,
                       image_col=args.image_col, label_col=args.label_col,
                       basename_lookup=args.basename_lookup)
    else:
        print("必须提供 --images_dir 或 --images_csv"); sys.exit(1)

    loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=args.num_workers,
                      shuffle=False,pin_memory=True)

    results=[]; t0=time.perf_counter(); total=len(dataset)
    sep=args.dup_sep

    with torch.no_grad():
        for batch in loader:
            pv=batch["pixel_values"].to(device=dev, dtype=dtype, non_blocking=True)
            paths=batch["path"]; labels=batch["label"]
            enc=model.get_encoder()(pixel_values=pv, return_dict=True).last_hidden_state
            enc=enc.to(device=dev, dtype=dtype)
            N=enc.shape[1]; Hp,Wp,_=guess_grid_from_N(N)

            # 带门控输出
            (ct_t, lg_t), (ct_b, lg_b), (g_t, g_b) = model.ctc_head(enc, Hp, Wp, return_gates=True)

            for i in range(pv.size(0)):
                # 逐行解码
                seq_t, s_t, m_t = ctc_beam_search(ct_t[:,i:i+1,:], beam=args.beam, blank_id=vocab.blank_id)
                seq_b, s_b, m_b = ctc_beam_search(ct_b[:,i:i+1,:], beam=args.beam, blank_id=vocab.blank_id)

                # 解码并移除重复分隔符（若训练使用了 |）
                txt_t = vocab.decode(seq_t).replace(sep, "") if sep else vocab.decode(seq_t)
                txt_b = vocab.decode(seq_b).replace(sep, "") if sep else vocab.decode(seq_b)

                # 每行置信度/非空白占比
                c_t, pr_t, _ = ctc_confidence(lg_t[i:i+1], blank_id=vocab.blank_id)
                c_b, pr_b, _ = ctc_confidence(lg_b[i:i+1], blank_id=vocab.blank_id)
                conf_t, conf_b = float(c_t[0]), float(c_b[0])
                prt, prb       = float(pr_t[0]), float(pr_b[0])

                # 合并 & 抑制重复
                merged_txt, line_mode = merge_two_lines(
                    txt_t, txt_b, conf_t, conf_b, prt, prb,
                    float(g_t[i].item()), float(g_b[i].item()),
                    min_len=2, pr_min=0.25, conf_min=0.65,
                    dup_sim_thr=0.6, gate_dom_thr=0.78
                )
                text_ctc = merged_txt

                # 回退门控：按“被保留行”的分数来判定
                if line_mode.startswith("top"):
                    used_conf, used_margin, used_pr = conf_t, float(m_t), prt
                elif line_mode.startswith("bot"):
                    used_conf, used_margin, used_pr = conf_b, float(m_b), prb
                else:
                    used_conf = min(conf_t, conf_b)
                    used_margin = float(min(m_t, m_b))
                    used_pr = max(prt, prb)

                do_fb = need_fallback(used_conf, used_margin, used_pr,
                                      thr_conf=args.conf_thr, thr_margin=args.margin_thr,
                                      pr_lo=args.pr_lo, pr_hi=args.pr_hi)

                results.append({
                    "path": paths[i],
                    "label": labels[i] if labels is not None else None,
                    "route": "ctc" if not do_fb else "ar_pending",
                    "pred_ctc": text_ctc,
                    "pred_ar": None,
                    "pred_final": text_ctc if not do_fb else None,
                    "conf_avg": used_conf,
                    "margin": used_margin,
                    "peak_ratio": used_pr,
                    "line_mode": line_mode,
                    "gate_top": float(g_t[i].item()),
                    "gate_bot": float(g_b[i].item()),
                })

        # ---- AR 回退 ----
        pend=[r for r in results if r["route"]=="ar_pending"]
        if pend:
            print(f"[INFO] AR 回退：{len(pend)}/{total} 张（{100.0*len(pend)/total:.2f}%）")
            B=args.batch_size
            for i in range(0,len(pend),B):
                sub=pend[i:i+B]
                ims=[Image.open(r["path"]).convert("RGB") for r in sub]
                pv=processor(images=ims, return_tensors="pt").pixel_values.to(device=dev, dtype=dtype)
                gen_ids=model.generate(pv, max_new_tokens=args.gen_max_new_tokens,
                                       num_beams=args.gen_beams, use_cache=True)
                texts=processor.batch_decode(gen_ids, skip_special_tokens=True)
                for r,txt in zip(sub,texts):
                    r["route"]="ar"; r["pred_ar"]=txt; r["pred_final"]=txt

    t1=time.perf_counter()
    print(f"[DONE] {total} images, time={t1-t0:.3f}s, avg={(t1-t0)/max(1,total):.5f}s/img")

    # 评估（若有标签）
    if any(r["label"] not in (None,"","nan") for r in results):
        print(f"[METRIC] CER(final): {np.mean([cer(str(r['pred_final'] or ''), str(r['label'] or '')) for r in results]):.4f}")

    # 保存
    out=os.path.abspath(args.out_csv); os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cols=["path","label","route","pred_ctc","pred_ar","pred_final","conf_avg","margin","peak_ratio","line_mode","gate_top","gate_bot"]
    pd.DataFrame([{k:r.get(k) for k in cols} for r in results]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {out}")

if __name__=="__main__":
    main()
