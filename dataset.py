#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, os, shutil, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------- 工具函数 ----------
def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def sanitize_label(s: str) -> str:
    """将标签转为文件名友好：大写 + 仅字母数字；需要保留连字符就把正则改为 [^A-Z0-9-]"""
    s = str(s).strip().upper()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s or "UNK"

def ensure_unique_filename(stem: str, used: Dict[str, int], ext: str = ".jpg") -> str:
    name = f"{stem}{ext}"
    if name not in used:
        used[name] = 1
        return name
    used[name] += 1
    return f"{stem}_{used[name]}{ext}"

def find_csv_candidates(src: Path) -> List[Path]:
    return sorted(list(src.glob("*.csv")) + list((src / "labels").glob("*.csv")))

def infer_mapping_from_csv(csv_path: Path, root: Path) -> List[Tuple[Path, str]]:
    """
    从 CSV 自动识别：文件列 和 标签列
    常见文件列名：['file','file_name','filename','image','img','path']
    常见标签列名：['label','plate','text','target','license','license_plate']
    """
    df = pd.read_csv(csv_path)
    file_cols = [c for c in df.columns if str(c).lower() in
                 ["file","file_name","filename","image","img","path"]]
    label_cols = [c for c in df.columns if str(c).lower() in
                  ["label","labels","plate","text","target","license","license_plate","plate_text"]]

    if not file_cols or not label_cols:
        # 尝试猜测：含有 "file" 或 ".jpg/.png" 的列当文件列；含有 "label"/"plate" 的列当标签列
        for c in df.columns:
            cl = str(c).lower()
            if not file_cols and ("file" in cl or "image" in cl or "img" in cl or "path" in cl):
                file_cols = [c]
            if not label_cols and ("label" in cl or "plate" in cl or "text" in cl or "license" in cl):
                label_cols = [c]

    if not file_cols or not label_cols:
        raise ValueError(f"无法在 {csv_path} 里推断文件列/标签列，请手动检查。列名：{list(df.columns)}")

    fcol, lcol = file_cols[0], label_cols[0]
    pairs: List[Tuple[Path, str]] = []
    for _, row in df.iterrows():
        rel = str(row[fcol]).strip()
        label = str(row[lcol]).strip()
        if not rel or not label:
            continue

        # 既支持相对路径也支持仅文件名
        cand = (root / rel)
        if not cand.exists():
            # 在根目录下按文件名搜索
            hits = list(root.rglob(Path(rel).name))
            if hits:
                cand = hits[0]
        if cand.exists() and is_image(cand):
            pairs.append((cand, label))
    return pairs

def infer_mapping_from_filenames(root: Path) -> List[Tuple[Path, str]]:
    pairs = []
    for p in root.rglob("*"):
        if is_image(p):
            label = sanitize_label(p.stem)
            pairs.append((p, label))
    return pairs

def copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError("mode must be copy|hardlink|symlink")

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="整理 Kaggle 车牌文本数据到 dataset/{train,val} + CSV（文件名=标签）")
    ap.add_argument("--src", default="./dataset/",required=True, help="Kaggle 数据集解压根目录")
    ap.add_argument("--out", default="./dataset", help="输出根目录（将创建 train/、val/ 及 CSV）")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="验证集占比")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["copy","hardlink","symlink"], default="copy", help="拷贝/硬链接/软链接")
    ap.add_argument("--ext", default=".jpg", help="输出图片扩展名（.jpg/.png），仅用于命名，不做重编码")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    out_train = out / "train"
    out_val = out / "val"
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: 建立 (image_path, label) 列表
    mapping: List[Tuple[Path, str]] = []
    csvs = find_csv_candidates(src)
    if csvs:
        # 优先使用 CSV
        for c in csvs:
            try:
                mapping.extend(infer_mapping_from_csv(c, src))
            except Exception as e:
                print(f"[WARN] 解析 {c.name} 失败：{e}")
    if not mapping:
        print("[INFO] 未找到可用 CSV，改为从文件名解析标签")
        mapping = infer_mapping_from_filenames(src)

    # 过滤与去重
    mapping = [(p, sanitize_label(lbl)) for p, lbl in mapping if lbl and is_image(p)]
    if not mapping:
        raise SystemExit("没有可用样本，请检查 --src 或 CSV 列名/图片扩展名。")

    # Step 2: 打乱并划分
    random.seed(args.seed)
    random.shuffle(mapping)
    n_total = len(mapping)
    n_val = max(1, int(n_total * args.val_ratio))
    val_set = mapping[:n_val]
    train_set = mapping[n_val:]

    # Step 3: 写入文件与 CSV
    def dump_split(samples: List[Tuple[Path, str]], split_dir: Path, csv_path: Path):
        used = {}
        rows = []
        for src_img, label in tqdm(samples, desc=f"Writing {split_dir.name}", ncols=100):
            name = ensure_unique_filename(label, used, ext=args.ext)
            dst = split_dir / name
            copy_or_link(src_img, dst, mode=args.mode)
            rows.append({"file_name": name, "label": label})
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"{split_dir.name}: {len(rows)} samples -> {csv_path}")

    dump_split(train_set, out_train, out / "train.csv")
    dump_split(val_set, out_val, out / "val.csv")
    print(f"Done. total={n_total}, train={len(train_set)}, val={len(val_set)}")
    print(f"结构示例：\n{out}/\n  train/\n  val/\n  train.csv\n  val.csv")

if __name__ == "__main__":
    main()
