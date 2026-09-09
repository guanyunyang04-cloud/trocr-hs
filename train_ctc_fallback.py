import os, json, math, argparse, random
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from PIL import Image
import pandas as pd
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

# -------------------- 实用工具 --------------------
def seed_everything(seed: int = 42):
    random.seed(seed); set_seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def guess_grid(N: int):
    s = int(math.sqrt(N))
    if s * s == N: return s, s, 0
    if (s * s + 1) == N: return s, s, 1  # 可能多一个 CLS
    if ((s+1)*(s+1)) - N <= 2: return s+1, s+1, 0
    return s, s, 0

# -------------------- 数据集 --------------------
class PlateDataset(Dataset):
    def __init__(self, csv_path: str, image_root: str, processor: TrOCRProcessor,
                 max_target_len: int = 32):
        self.df = pd.read_csv(csv_path)
        assert {'image_path','label'}.issubset(set(self.df.columns))
        self.root = image_root
        self.processor = processor
        self.max_target_len = max_target_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row['image_path'])
        text = str(row['label'])
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]

        tok = self.processor.tokenizer
        enc = tok(text, max_length=self.max_target_len, padding="max_length",
                  truncation=True, return_tensors="pt")
        labels = enc.input_ids[0]
        labels[enc.attention_mask[0]==0] = -100
        return {"pixel_values": pixel_values, "labels": labels, "text": text}

# -------------------- CTC 字符表 --------------------
class CharVocab:
    def __init__(self, chars: str):
        chars = sorted(set(chars))
        self.chars = ["<blank>"] + chars
        self.stoi = {c:i for i,c in enumerate(self.chars)}
        self.itos = {i:c for c,i in self.stoi.items()}
        self.blank_id = 0
    @staticmethod
    def from_datasets(dfs: List[pd.DataFrame], col="label"):
        charset = set()
        for df in dfs:
            for t in df[col].astype(str).tolist():
                charset.update(list(t.strip()))
        return CharVocab("".join(sorted(charset)))
    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]
    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids if i != self.blank_id)

# -------------------- CTC 头（行聚合 + 线性） --------------------
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

# -------------------- CTC beam 与置信度 --------------------
from collections import defaultdict
def ctc_beam_search(log_probs: torch.Tensor, beam=5, blank_id=0, topk_each_step=10):
    T,B,C = log_probs.shape
    assert B==1
    beams = {(): (0.0, float('-inf'))}
    for t in range(T):
        lp = log_probs[t,0]
        nxt = defaultdict(lambda: (float('-inf'), float('-inf')))
        for pref,(pb,pnb) in beams.items():  # blank
            nb = nxt[pref]
            nb = (torch.logsumexp(torch.tensor([nb[0], pb+lp[blank_id], pnb+lp[blank_id]]),0).item(), nb[1])
            nxt[pref] = nb
        for c in torch.topk(lp, k=min(topk_each_step, C)).indices.tolist():  # char
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

def ctc_confidence_from_logits(logits: torch.Tensor, blank_id=0):
    probs = logits.softmax(-1)
    maxp, maxi = probs.max(-1)
    blank = (maxi==blank_id).float()
    conf_avg   = maxp.mean(dim=1)            # (B,)
    peak_ratio = 1.0 - blank.mean(dim=1)     # (B,)
    entropy    = -(probs * (probs.clamp_min(1e-9).log())).sum(-1).mean(1)
    return conf_avg, peak_ratio, entropy

# -------------------- 多任务 Trainer --------------------
@dataclass
class MTConfig:
    stage: str = "warmup-ctc"     # "warmup-ctc" or "joint"
    lambda_ctc: float = 0.7
    unfreeze_encoder_last: int = 0
    freeze_decoder: bool = True

class MTTrainer(Trainer):
    def __init__(self, *args, mtcfg: MTConfig, ctc_vocab: CharVocab, Hp: int, Wp: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.mtcfg = mtcfg
        self.ctc_vocab = ctc_vocab
        self.Hp, self.Wp = Hp, Wp
        self.ctc_loss = nn.CTCLoss(blank=ctc_vocab.blank_id, zero_infinity=True)

    def compute_loss(self, model: VisionEncoderDecoderModel, inputs: Dict[str, Any], return_outputs=False):
        pixel_values = inputs["pixel_values"]
        labels = inputs["labels"]

        if self.mtcfg.stage == "warmup-ctc":
            enc_out = model.get_encoder()(pixel_values=pixel_values, return_dict=True).last_hidden_state
            if enc_out.shape[1] == (self.Hp*self.Wp + 1):
                enc_out = enc_out[:, 1:, :]  # drop CLS
            logp, _ = model.ctc_head(enc_out, self.Hp, self.Wp)  # (T,B,C)

            texts: List[str] = inputs["text"]
            tgt_list = [torch.tensor(self.ctc_vocab.encode(t), dtype=torch.long, device=logp.device) for t in texts]
            target_lengths = torch.tensor([t.numel() for t in tgt_list], dtype=torch.long, device=logp.device)
            if target_lengths.max()==0:
                loss_ctc = torch.tensor(0.0, device=logp.device)
            else:
                targets_flat = torch.cat([t for t in tgt_list if t.numel()>0], dim=0)
                input_lengths = torch.full((pixel_values.size(0),), fill_value=logp.shape[0], dtype=torch.long, device=logp.device)
                loss_ctc = self.ctc_loss(logp, targets_flat, input_lengths, target_lengths)
            return (loss_ctc, {"loss_ctc":loss_ctc}) if return_outputs else loss_ctc

        # joint：AR + CTC
        outputs = model(pixel_values=pixel_values, labels=labels, output_hidden_states=False, return_dict=True)
        loss_ar = outputs.loss
        enc_last = outputs.encoder_last_hidden_state
        if enc_last.shape[1] == (self.Hp*self.Wp + 1):
            enc_last = enc_last[:, 1:, :]
        logp, _ = model.ctc_head(enc_last, self.Hp, self.Wp)

        texts: List[str] = inputs["text"]
        tgt_list = [torch.tensor(self.ctc_vocab.encode(t), dtype=torch.long, device=logp.device) for t in texts]
        target_lengths = torch.tensor([t.numel() for t in tgt_list], dtype=torch.long, device=logp.device)
        if target_lengths.max()==0:
            loss_ctc = torch.tensor(0.0, device=logp.device)
        else:
            targets_flat = torch.cat([t for t in tgt_list if t.numel()>0], dim=0)
            input_lengths = torch.full((pixel_values.size(0),), fill_value=logp.shape[0], dtype=torch.long, device=logp.device)
            loss_ctc = self.ctc_loss(logp, targets_flat, input_lengths, target_lengths)

        loss = self.mtcfg.lambda_ctc * loss_ctc + (1.0 - self.mtcfg.lambda_ctc) * loss_ar
        return (loss, {"loss_ctc":loss_ctc, "loss_ar":loss_ar}) if return_outputs else loss

# -------------------- 冻结/解冻 --------------------
def freeze_all_but_ctc(model: VisionEncoderDecoderModel):
    for p in model.parameters(): p.requires_grad = False
    for p in model.ctc_head.parameters(): p.requires_grad = True

def unfreeze_encoder_last_k(model: VisionEncoderDecoderModel, k: int):
    if k <= 0: return
    enc = model.get_encoder()
    # 尝试常见 ViT/BEiT 命名
    if hasattr(enc, "encoder") and hasattr(enc.encoder, "layer"):
        layers = enc.encoder.layer
    elif hasattr(enc, "vit") and hasattr(enc.vit, "encoder"):
        layers = enc.vit.encoder.layer
    else:
        print("[WARN] 未识别 encoder 层结构，跳过选择性解冻")
        return
    for p in enc.parameters(): p.requires_grad = False
    for layer in layers[-k:]:
        for p in layer.parameters(): p.requires_grad = True

def maybe_freeze_decoder(model: VisionEncoderDecoderModel, freeze: bool = True):
    if not freeze: return
    dec = model.get_decoder()
    for p in dec.parameters(): p.requires_grad = False

# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stage", choices=["warmup-ctc","joint"], default="warmup-ctc")
    ap.add_argument("--lambda_ctc", type=float, default=0.7)
    ap.add_argument("--unfreeze_encoder_last", type=int, default=0)
    ap.add_argument("--freeze_decoder", action="store_true", default=False)
    ap.add_argument("--per_device_train_batch_size", type=int, default=32)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=32)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=5e-4)
    ap.add_argument("--num_train_epochs", type=int, default=2)
    ap.add_argument("--max_target_len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true", default=False)
    ap.add_argument("--fp16", action="store_true", default=False)
    ap.add_argument("--dataloader_num_workers", type=int, default=8)
    ap.add_argument("--deepspeed", type=str, default=None)
    ap.add_argument("--ddp_find_unused_parameters", type=str, default=None,
                    help="true/false；warmup 建议 true，joint 可 false")
    ap.add_argument("--ctc_chars", type=str, default="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
                    help="CTC 字符集合（不含 <blank>）")
    args = ap.parse_args()

    seed_everything(args.seed)

    # 1) 加载基线模型 & 处理器（保持 CPU，设备由 Trainer 管）
    processor = TrOCRProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

    # 2) CTC 词表（固定或由数据推断）
    # df_tr = pd.read_csv(args.train_csv); df_va = pd.read_csv(args.val_csv)
    # ctc_vocab = CharVocab.from_datasets([df_tr, df_va], col="label")
    ctc_vocab = CharVocab(args.ctc_chars)

    # 3) 猜测 encoder 网格尺寸 (H',W')（在 CPU 上做一次 dummy 前向）
    with torch.no_grad():
        dummy = processor(images=Image.new("RGB",(384,384),(0,0,0)), return_tensors="pt").pixel_values
        enc = model.get_encoder()(pixel_values=dummy, return_dict=True).last_hidden_state
        N = enc.shape[1]
    Hp, Wp, drop = guess_grid(N)
    if drop==1:
        print(f"[CTC] Detected possible CLS: N={N} -> use N-1={N-1} as grid {Hp}x{Wp}")
    print(f"[CTC] Grid guess: H'={Hp}, W'={Wp}, N={N}")

    # 4) 把 CTC 头挂到模型（Trainer 会把整个 model 分发到多卡）
    d_model = model.config.encoder.hidden_size
    model.ctc_head = CTCHead(d_in=d_model, vocab_size=len(ctc_vocab.chars), blank_id=ctc_vocab.blank_id)

    # 5) 冻结/解冻策略
    if args.stage == "warmup-ctc":
        freeze_all_but_ctc(model)
        print("[Train] warmup-ctc: 仅训练 CTC 头")
    else:
        unfreeze_encoder_last_k(model, args.unfreeze_encoder_last)
        maybe_freeze_decoder(model, args.freeze_decoder)
        print(f"[Train] joint: lambda_ctc={args.lambda_ctc}, "
              f"unfreeze_encoder_last={args.unfreeze_encoder_last}, "
              f"freeze_decoder={args.freeze_decoder}")

    # 6) 数据集
    train_set = PlateDataset(args.train_csv, args.image_root, processor, args.max_target_len)
    val_set   = PlateDataset(args.val_csv,   args.image_root, processor, args.max_target_len)

    # 7) 训练参数（注意：设备/分布式由 Trainer 管理）
    # 解析 ddp_find_unused_parameters
    ddp_fup = None
    if args.ddp_find_unused_parameters is not None:
        ddp_fup = args.ddp_find_unused_parameters.lower() == "true"

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=ddp_fup,
    )

    # 8) 多任务 Trainer
    mtcfg = MTConfig(
        stage=args.stage,
        lambda_ctc=args.lambda_ctc,
        unfreeze_encoder_last=args.unfreeze_encoder_last,
        freeze_decoder=args.freeze_decoder
    )

    trainer = MTTrainer(
        model=model,
        args=targs,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=default_data_collator,
        mtcfg=mtcfg,
        ctc_vocab=ctc_vocab,
        Hp=Hp, Wp=Wp,
        tokenizer=processor.tokenizer,
    )

    # 9) 训练
    trainer.train()

    # 10) 保存（包含 ctc_head 权重 + 词表）
    trainer.save_model(args.output_dir)
    with open(os.path.join(args.output_dir, "ctc_vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"chars": ctc_vocab.chars, "blank_id": ctc_vocab.blank_id}, f, ensure_ascii=False, indent=2)
    print("[Done] Saved to", args.output_dir)

if __name__ == "__main__":
    main()

