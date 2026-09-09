import argparse
import os
import sys
import csv
import glob
import shutil
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image, UnidentifiedImageError, ImageDraw, ImageFont
from tqdm import tqdm
import yaml

from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def safe_import_convert_to_delpy():
    """与训练脚本一致：尽量从 deploy.utils 引入 convert_to_delpy，可缺省。"""
    try:
        from deploy.utils import convert_to_delpy  # type: ignore
        return convert_to_delpy
    except Exception:
        def _noop(model):
            return model
        return _noop


def load_cfg(cfg_path: Optional[str]):
    if not cfg_path:
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(root: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[str]:
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(root, "**", f"*{e}"), recursive=True))
    return sorted(paths)


def prefix_label_from_name(p: str) -> str:
    """文件名首个下划线前缀作为预标注。"""
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    return stem.split("_", 1)[0]


def normalize_text(s: str, case_insensitive: bool = True, alnum_only: bool = False) -> str:
    t = s.strip()
    if alnum_only:
        t = "".join(ch for ch in t if ch.isalnum())
    if case_insensitive:
        t = t.upper()
    return t


@torch.inference_mode()
def run_infer(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    device: torch.device,
    image_paths: List[str],
    batch_size: int = 8,
    fp16: bool = False,
) -> List[str]:
    """逐张推理（与训练脚本一致：processor(images=...).pixel_values -> model.generate）。"""
    preds: List[str] = []
    model.eval()
    for i in tqdm(range(0, len(image_paths), batch_size), desc="TroCR inference"):
        batch_files = image_paths[i : i + batch_size]
        images = []
        for f in batch_files:
            try:
                img = Image.open(f).convert("RGB")
            except (UnidentifiedImageError, FileNotFoundError):
                # 放一个白图占位，避免中断
                img = Image.new("RGB", (384, 384), (255, 255, 255))
            images.append(img)

        inputs = processor(images=images, return_tensors="pt").to(device)
        if fp16 and device.type == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()

        gen_ids = model.generate(**inputs)
        texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
        texts = [t.strip() for t in texts]
        preds.extend(texts)
    return preds


def main():
    parser = argparse.ArgumentParser(description="TroCR 推理筛选（按训练脚本风格）")
    parser.add_argument("--image-dir", required=True, help="输入图片根目录（递归遍历）")
    parser.add_argument("--output-dir", required=True, help="一致图片复制到此目录")
    parser.add_argument("--cfg", type=str, default=None, help="训练用 YAML（读取 model_config.model_name）")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace 模型名，如 microsoft/trocr-base-printed")
    parser.add_argument("--checkpoint", type=str, default=None, help="本地训练输出目录（Trainer 保存的权重）")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--case-insensitive", action="store_true", default=True)
    parser.add_argument("--alnum-only", action="store_true", default=False)
    parser.add_argument("--preserve-structure", action="store_true", help="复制时保留子目录结构")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # ---- 解析模型来源（优先级：checkpoint > model > cfg）----
    model_name_or_path = None
    if args.checkpoint and os.path.isdir(args.checkpoint):
        model_name_or_path = args.checkpoint
        print(f"[INFO] 使用 checkpoint: {args.checkpoint}")
    elif args.model:
        model_name_or_path = args.model
        print(f"[INFO] 使用模型: {args.model}")
    else:
        cfg = load_cfg(args.cfg)
        if not cfg or "model_config" not in cfg or "model_name" not in cfg["model_config"]:
            print("[ERROR] 未提供 --checkpoint / --model，且 cfg 中没有 model_config.model_name")
            sys.exit(1)
        model_name_or_path = cfg["model_config"]["model_name"]
        print(f"[INFO] 使用 cfg.model_config.model_name: {model_name_or_path}")

    # ---- 加载 Processor & Model（与训练脚本一致）----
    convert_to_delpy = safe_import_convert_to_delpy()
    try:
        processor = TrOCRProcessor.from_pretrained(model_name_or_path)
    except Exception as e:
        # 某些 checkpoint 目录不含 tokenizer 配置，回退到基础模型名（若提供）
        if args.model and args.checkpoint:
            print(f"[WARN] 从 checkpoint 加载 processor 失败，尝试从 --model 加载：{e}")
            processor = TrOCRProcessor.from_pretrained(args.model)
        else:
            raise

    model = VisionEncoderDecoderModel.from_pretrained(
        model_name_or_path, ignore_mismatched_sizes=True
    )
    model = convert_to_delpy(model)
    model.to(device)
    print(f"[INFO] generation_config = {dict(model.generation_config.to_dict())}")

    # ---- 收集图片 ----
    image_paths = list_images(args.image_dir)
    if not image_paths:
        print("[ERROR] 未找到图片")
        sys.exit(1)
    print(f"[INFO] 待处理图片数: {len(image_paths)}")

    # ---- 推理 ----
    preds = run_infer(
        model=model,
        processor=processor,
        device=device,
        image_paths=image_paths,
        batch_size=args.batch_size,
        fp16=args.fp16,
    )

    # ---- 对比 + 复制 + 记录 ----
    csv_path = os.path.join(args.output_dir, "matched.csv")
    matched = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prefix_label", "trocr_pred", "matched_eq"])

        for img_path, pred_text in zip(image_paths, preds):
            pref = prefix_label_from_name(img_path)

            a = normalize_text(pref, case_insensitive=args.case_insensitive, alnum_only=args.alnum_only)
            b = normalize_text(pred_text, case_insensitive=args.case_insensitive, alnum_only=args.alnum_only)

            is_eq = (a == b)
            writer.writerow([img_path, pref, pred_text, int(is_eq)])

            if is_eq:
                if args.preserve_structure:
                    # 保留相对目录结构
                    rel = os.path.relpath(img_path, args.image_dir)
                    dst = os.path.join(args.output_dir, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                else:
                    # 扁平复制；重名则加序号
                    base = os.path.basename(img_path)
                    dst = os.path.join(args.output_dir, base)
                    if os.path.exists(dst):
                        stem, ext = os.path.splitext(base)
                        k = 1
                        while True:
                            alt = os.path.join(args.output_dir, f"{stem}_{k}{ext}")
                            if not os.path.exists(alt):
                                dst = alt
                                break
                            k += 1
                shutil.copy2(img_path, dst)
                matched += 1

    print(f"[DONE] 匹配成功 {matched} / {len(image_paths)}")
    print(f"[INFO] 记录表保存：{csv_path}")
    print(f"[INFO] 已将匹配图片复制到：{args.output_dir}")


if __name__ == "__main__":
    main()
