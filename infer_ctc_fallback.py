import os, sys, math, time, argparse, glob, json
from typing import List, Tuple, Dict, Any
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# -------------------- 小工具 --------------------
def device_pick():
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
        for j in range(1,n+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1] + (0 if ref[i-1]==pred[j-1] else 1))
    return dp[m][n] / max(1, m)

# -------------------- CTC 词表 --------------------
class CharVocab:
    def __init__(self, chars: str, blank_id: int = 0):
        self.chars = ["<blank>"] + sorted(set(chars))
        self.stoi = {c:i for i,c in enumerate(self.chars)}
        self.itos = {i:c for c,i in self.stoi.items()}
        self.blank_id = blank_id
    @staticmethod
    def load(model_dir: str, fallback_chars: str = None):
        vp = os.path.join(model_dir, "ctc_vocab.json")
        if os.path.exists(vp):
            j = json.load(open(vp, "r", encoding="utf-8"))
            chars = "".join([c for c in j["chars"] if c != "<blank>"])
            return CharVocab(chars, blank_id=j.get("blank_id", 0))
        if fallback_chars:
            print(f"[WARN] ctc_vocab.json 未找到，改用 --ctc_chars")
            return CharVocab(fallback_chars)
        raise FileNotFoundError("未找到 ctc_vocab.json，且未指定 --ctc_chars")

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]
    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids if i != self.blank_id)

# -------------------- CTC 头（用于装回权重；结构需与训练一致） --------------------
class CTCHead(nn.Module):
    def __init__(self, d_in: int, vocab_size: int, blank_id: int = 0):
        super().__init__()
        self.dw1 = nn.Conv1d(d_in, d_in, 3, padding=1, groups=d_in)
        self.pw  = nn.Linear(d_in, d_in, bias=False)
        self.dw2 = nn.Conv1d(d_in, d_in, 3, padding=1, groups=d_in)
        self.cls = nn.Linear(d_in, vocab_size)
        self.blank_id = blank_id
    def forward(self, enc_tokens: torch.Tensor, Hp: int, Wp: int):
        B,N,D = enc_tokens.shape
        target = Hp*Wp
        extra = N - target
        if extra in (1,2):    # 兼容 CLS / CLS+DISTILL
            enc_tokens = enc_tokens[:, extra:, :]
            N -= extra
        assert N == target, f"CTCHead: N={N} != H'W'={target}"
        x = enc_tokens.view(B, Hp, Wp, D).mean(dim=1)  # (B, W', D)
        res = x
        x = self.pw(x)
        x = x.transpose(1,2)                           # (B, D, W')
        x = F.gelu(self.dw1(x))
        x = F.gelu(self.dw2(x))
        x = x.transpose(1,2) + res                     # (B, W', D)
        logits = self.cls(x)                           # (B, W', C)
        logp = logits.log_softmax(dim=-1).transpose(0,1)  # (T=W', B, C)
        return logp, logits

# -------------------- CTC beam & 置信度 --------------------
from collections import defaultdict
def ctc_beam_search(log_probs: torch.Tensor, beam=5, blank_id=0, topk_each_step=10):
    T,B,C = log_probs.shape
    assert B==1
    beams = {(): (0.0, float('-inf'))}
    for t in range(T):
        lp = log_probs[t,0]
        nxt = defaultdict(lambda: (float('-inf'), float('-inf')))
        # blank
        for pref,(pb,pnb) in beams.items():
            nb = nxt[pref]
            nb = (torch.logsumexp(torch.tensor([nb[0], pb+lp[blank_id], pnb+lp[blank_id]]),0).item(), nb[1])
            nxt[pref] = nb
        # chars
        topk = torch.topk(lp, k=min(topk_each_step, C)).indices.tolist()
        for c in topk:
            if c==blank_id: continue
            for pref,(pb,pnb) in beams.items():
                if len(pref)>0 and c==pref[-1]:
                    nb = nxt[pref]
                    nb = (nb[0], torch.logsumexp(torch.tensor([nb[1], pb+lp[c]]),0).item())
                    nxt[pref] = nb
                else:
                    newp = pref+(c,)
                    nb = nxt[newp]
                    nb = (nb[0], torch.logsumexp(torch.tensor([nb[1], pb+lp[c], pnb+lp[c]]),0).item())
                    nxt[newp] = nb
        beams = dict(sorted(nxt.items(),
                            key=lambda kv: torch.logsumexp(torch.tensor(kv[1]),0).item(),
                            reverse=True)[:beam])
    scored = [(list(p), torch.logsumexp(torch.tensor(v),0).item()) for p,v in beams.items()]
    scored.sort(key=lambda x:x[1], reverse=True)
    seq1,s1 = (scored[0][0], scored[0][1]) if scored else ([], -1e9)
    seq2,s2 = (scored[1][0], scored[1][1]) if len(scored)>1 else (seq1, s1-1e3)
    margin = (s1 - s2) / max(1,len(seq1))
    return seq1, s1, margin

def ctc_confidence(logits: torch.Tensor, blank_id=0):
    probs = logits.softmax(-1)
    maxp, maxi = probs.max(-1)
    blank = (maxi==blank_id).float()
    conf_avg   = maxp.mean(dim=1)            # (B,)
    peak_ratio = 1.0 - blank.mean(dim=1)     # (B,)
    entropy    = -(probs * (probs.clamp_min(1e-9).log())).sum(-1).mean(1)
    return conf_avg, peak_ratio, entropy

def need_fallback(conf_avg, margin, peak_ratio, thr_conf=0.90, thr_margin=0.30, pr_lo=0.35, pr_hi=0.85):
    return (conf_avg < thr_conf) or (margin < thr_margin) or not (pr_lo <= peak_ratio <= pr_hi)

# -------------------- 数据集 --------------------
class ImgCSV(Dataset):
    def __init__(self, csv_path, image_root, processor, image_col=None, label_col=None, basename_lookup=False, exts=".jpg,.jpeg,.png,.bmp"):
        self.df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        self.df.columns = [c.strip() for c in self.df.columns]
        self.root = image_root
        self.proc = processor
        self.basename_lookup = basename_lookup
        self.exts = tuple([e.strip().lower() for e in exts.split(",") if e.strip()])
        # 列名探测
        def norm(s): return s.strip().lower().replace(" ", "").replace("_","").replace("-","")
        inv = {}
        for c in self.df.columns: inv.setdefault(norm(c), c)
        img_candidates = [image_col] if image_col else ["imagepath","filepath","path","filename","file","img","imagename","imgpath","imgfile"]
        lab_candidates = [label_col] if label_col else ["label","text","gt","target","transcript","labeltext"]
        self.ic = next((inv[x] for x in img_candidates if x and x in inv), None)
        self.lc = next((inv[x] for x in lab_candidates if x and x in inv), None)
        if self.ic is None: raise ValueError(f"CSV 中找不到图片列；可用 --image_col 指定。现有列：{list(self.df.columns)}")

        # basename 索引
        self.name2path = None
        if basename_lookup:
            self.name2path = {}
            for dirpath,_,files in os.walk(self.root):
                for f in files:
                    fl = f.lower()
                    if not fl.endswith(self.exts): continue
                    self.name2path.setdefault(fl, os.path.join(dirpath,f))

    def __len__(self): return len(self.df)
    def _resolve(self, val):
        s = str(val).strip()
        if os.path.isabs(s): return s
        if "/" in s or "\\" in s: return os.path.join(self.root, s)
        if self.basename_lookup and self.name2path is not None:
            p = self.name2path.get(s.lower())
            if p is None: raise FileNotFoundError(f"basename {s} 未在 {self.root} 下找到")
            return p
        return os.path.join(self.root, s)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = self._resolve(row[self.ic])
        lab = str(row[self.lc]) if self.lc is not None else None
        im = Image.open(p).convert("RGB")
        pv = self.proc(images=im, return_tensors="pt").pixel_values[0]
        return {"pixel_values": pv, "path": p, "label": lab}

class ImgDir(Dataset):
    def __init__(self, images_dir, processor, exts=(".jpg",".jpeg",".png",".bmp")):
        self.paths = [p for e in exts for p in glob.glob(os.path.join(images_dir, f"**/*{e}"), recursive=True)]
        if not self.paths: raise FileNotFoundError(f"目录下未找到图片：{images_dir}")
        self.proc = processor
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        im = Image.open(p).convert("RGB")
        pv = self.proc(images=im, return_tensors="pt").pixel_values[0]
        return {"pixel_values": pv, "path": p, "label": None}

# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    # 输入方式二选一
    ap.add_argument("--images_dir", type=str, default=None)
    ap.add_argument("--images_csv", type=str, default=None)
    ap.add_argument("--image_root", type=str, default=".")
    ap.add_argument("--image_col", type=str, default=None)
    ap.add_argument("--label_col", type=str, default=None)
    ap.add_argument("--basename_lookup", action="store_true", default=False)

    ap.add_argument("--ctc_chars", type=str, default=None, help="若无 ctc_vocab.json，用此固定字符集（不含 <blank>）")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)

    # 门控阈值
    ap.add_argument("--conf_thr", type=float, default=0.90)
    ap.add_argument("--margin_thr", type=float, default=0.30)
    ap.add_argument("--pr_lo", type=float, default=0.35)
    ap.add_argument("--pr_hi", type=float, default=0.85)

    # AR 生成参数
    ap.add_argument("--gen_max_new_tokens", type=int, default=24)
    ap.add_argument("--gen_beams", type=int, default=1)

    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    dev = device_pick()
    print(f"[INFO] device = {dev}")

    # 1) 加载模型/处理器/词表
    processor = TrOCRProcessor.from_pretrained(args.model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir)
    model.eval().to(dev)

    ctc_vocab = CharVocab.load(args.model_dir, fallback_chars=args.ctc_chars)

    # 如果模型里没有 ctc_head，尝试构造并从权重文件装回
    if not hasattr(model, "ctc_head"):
        model.ctc_head = CTCHead(model.config.encoder.hidden_size, len(ctc_vocab.chars), blank_id=ctc_vocab.blank_id).to(dev)
        # 尝试从 bin 里加载 ctc_head.* 权重
        sd_path_bin = os.path.join(args.model_dir, "pytorch_model.bin")
        loaded = False
        if os.path.exists(sd_path_bin):
            sd = torch.load(sd_path_bin, map_location="cpu")
            sub = {k.replace("ctc_head.", ""): v for k,v in sd.items() if k.startswith("ctc_head.")}
            if sub:
                miss,unexp = model.ctc_head.load_state_dict(sub, strict=False)
                print(f"[INFO] loaded CTC head weights from bin (missing={len(miss)}, unexpected={len(unexp)})")
                loaded = True
        if not loaded:
            print("[WARN] 未能从模型权重中恢复 ctc_head，使用随机初始化（效果会差）")

    # 2) 构造数据集
    if args.images_dir:
        dataset = ImgDir(args.images_dir, processor)
    elif args.images_csv:
        dataset = ImgCSV(args.images_csv, args.image_root, processor,
                         image_col=args.image_col, label_col=args.label_col,
                         basename_lookup=args.basename_lookup)
    else:
        print("必须提供 --images_dir 或 --images_csv 之一"); sys.exit(1)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    results = []
    t0 = time.perf_counter()
    total = len(dataset)

    with torch.no_grad():
        # 先整批跑 CTC
        for batch in loader:
            pv = batch["pixel_values"].to(dev, non_blocking=True)
            paths = batch["path"]
            labels = batch["label"]  # 可能全是 None

            # 编码
            enc = model.get_encoder()(pixel_values=pv, return_dict=True).last_hidden_state  # (B, N, D)
            N = enc.shape[1]
            s = int(round((N if N<=1024 else 576) ** 0.5))
            # 默认 384/patch=16 => 24；若用别的尺寸，可根据 N 自适应
            Hp = Wp = int(math.sqrt(N if N in (s*s, s*s+1, s*s+2) else s*s))
            logp, logits = model.ctc_head(enc, Hp, Wp)  # CTCHead 内部会裁 CLS/distill

            # 逐样本 beam + 置信度
            for i in range(pv.size(0)):
                lp = logp[:, i:i+1, :].contiguous()
                seq, s1, margin = ctc_beam_search(lp, beam=5, blank_id=ctc_vocab.blank_id)
                text_ctc = ctc_vocab.decode(seq)

                conf_avg, peak_ratio, entropy = ctc_confidence(logits[i:i+1], blank_id=ctc_vocab.blank_id)
                conf = float(conf_avg[0].item()); pr = float(peak_ratio[0].item())

                do_fallback = need_fallback(conf, float(margin), pr, args.conf_thr, args.margin_thr, args.pr_lo, args.pr_hi)

                results.append({
                    "path": paths[i],
                    "label": labels[i] if labels is not None else None,
                    "route": "ctc" if not do_fallback else "ar_pending",
                    "pred_ctc": text_ctc,
                    "pred_final": text_ctc if not do_fallback else None,
                    "conf_avg": conf,
                    "margin": float(margin),
                    "peak_ratio": pr,
                })

        # 收集需要回退的样本，分批跑 AR generate
        pending = [r for r in results if r["route"]=="ar_pending"]
        if pending:
            print(f"[INFO] AR 回退：{len(pending)}/{total} 张（{100.0*len(pending)/total:.2f}%）")
            # 为简洁，第二次从磁盘读图再跑一次模型（回退占比通常很低）
            # 你也可以缓存 pixel_values 以避免重复 encode
            batch_paths = [r["path"] for r in pending]
            B = args.batch_size
            for i in range(0, len(batch_paths), B):
                sub = batch_paths[i:i+B]
                ims = [Image.open(p).convert("RGB") for p in sub]
                pv = processor(images=ims, return_tensors="pt").pixel_values.to(dev)
                gen_ids = model.generate(pv, max_new_tokens=args.gen_max_new_tokens, num_beams=args.gen_beams, use_cache=True)
                texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
                # 写回
                for pth, txt in zip(sub, texts):
                    for r in results:
                        if r["path"]==pth and r["route"]=="ar_pending":
                            r["route"]="ar"
                            r["pred_ar"]=txt
                            r["pred_final"]=txt
                            break

    t1 = time.perf_counter()
    print(f"[DONE] {total} images, time={t1-t0:.3f}s, avg={((t1-t0)/max(1,total)):.4f}s/img")

    # 计算 CER（若有标签）
    have_label = any(r["label"] not in (None, "", "nan") for r in results)
    if have_label:
        cers = []
        for r in results:
            if r["label"] not in (None, "", "nan"):
                cers.append(cer(r["pred_final"] or "", str(r["label"])))
        if cers:
            print(f"[METRIC] CER (final): {np.mean(cers):.4f}")

    # 写出 CSV
    cols = ["path","label","route","pred_ctc","pred_ar","pred_final","conf_avg","margin","peak_ratio"]
    df = pd.DataFrame([{k: r.get(k, None) for k in cols} for r in results])
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {args.out_csv}")

if __name__ == "__main__":
    main()
