import argparse
import os
from typing import List, Optional

import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def safe_import_convert_to_delpy():
    """Use deploy-time model patching when available; otherwise keep evaluation portable."""
    try:
        from deploy.utils import convert_to_delpy  # type: ignore
        return convert_to_delpy
    except Exception as exc:
        print(f"[WARN] deploy patch is unavailable, skip convert_to_delpy: {exc}")

        def _noop(model):
            return model

        return _noop


def cer(pred: str, ref: str) -> float:
    m, n = len(ref), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (0 if ref[i - 1] == pred[j - 1] else 1),
            )
    return dp[m][n] / max(1, m)


def normalize_text(text: str, case_insensitive: bool = True, alnum_only: bool = False) -> str:
    out = str(text).strip()
    if case_insensitive:
        out = out.upper()
    out = out.replace(" ", "").replace("-", "")
    if alnum_only:
        out = "".join(ch for ch in out if ch.isalnum())
    return out


def resolve_col(df: pd.DataFrame, explicit: Optional[str], candidates: List[str], what: str) -> str:
    raw = list(df.columns)

    def norm(s: str) -> str:
        return s.strip().lower().replace(" ", "").replace("_", "").replace("-", "")

    norm_to_raw = {}
    for col in raw:
        key = norm(col)
        if key not in norm_to_raw:
            norm_to_raw[key] = col

    if explicit:
        if explicit in raw:
            return explicit
        explicit_norm = norm(explicit)
        if explicit_norm in norm_to_raw:
            return norm_to_raw[explicit_norm]
        raise ValueError(f"--{what}='{explicit}' not found; csv cols={raw}")

    for cand in candidates:
        if cand in raw:
            return cand
        cand_norm = norm(cand)
        if cand_norm in norm_to_raw:
            return norm_to_raw[cand_norm]

    raise ValueError(f"could not infer {what}; csv cols={raw}")


def resolve_image_path(image_dir: str, image_value: str) -> str:
    image_value = str(image_value).strip()
    if os.path.isabs(image_value):
        return image_value
    return os.path.join(image_dir, image_value)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a TrOCR model on a labeled CSV dataset.")
    parser.add_argument("--model", required=True, help="Model directory or Hugging Face model name")
    parser.add_argument("--csv", required=True, help="CSV file containing image path/name and label columns")
    parser.add_argument("--image-dir", required=True, help="Image root directory")
    parser.add_argument("--image-col", type=str, default=None, help="CSV image column name")
    parser.add_argument("--label-col", type=str, default=None, help="CSV label column name")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N rows; 0 means all rows")
    parser.add_argument("--case-insensitive", action="store_true", default=True)
    parser.add_argument("--alnum-only", action="store_true", default=False)
    parser.add_argument("--out-csv", type=str, default=None, help="Optional path to save per-sample predictions")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    image_col = resolve_col(
        df,
        args.image_col,
        ["file_name", "image_path", "image", "img", "path", "filename", "file"],
        "image_col",
    )
    label_col = resolve_col(
        df,
        args.label_col,
        ["label", "text", "plate", "target", "license_plate"],
        "label_col",
    )
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] rows={len(df)} image_col='{image_col}' label_col='{label_col}'")

    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    model = safe_import_convert_to_delpy()(model)
    model = model.to(device).eval()

    rows = df.to_dict("records")
    results = []
    exact_raw = 0
    exact_norm = 0
    cers = []

    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            images = []
            labels = []
            image_names = []

            for row in batch:
                image_name = row[image_col]
                label = row[label_col]
                image_path = resolve_image_path(args.image_dir, image_name)
                images.append(Image.open(image_path).convert("RGB"))
                labels.append(str(label))
                image_names.append(image_name)

            pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            preds = [pred.strip() for pred in processor.batch_decode(generated_ids, skip_special_tokens=True)]

            for image_name, label, pred in zip(image_names, labels, preds):
                raw_equal = pred.strip() == label.strip()
                pred_norm = normalize_text(pred, args.case_insensitive, args.alnum_only)
                label_norm = normalize_text(label, args.case_insensitive, args.alnum_only)
                norm_equal = pred_norm == label_norm

                exact_raw += int(raw_equal)
                exact_norm += int(norm_equal)
                cers.append(cer(pred_norm, label_norm))
                results.append(
                    {
                        "image_name": image_name,
                        "label": label,
                        "pred": pred,
                        "pred_norm": pred_norm,
                        "label_norm": label_norm,
                        "exact_raw": int(raw_equal),
                        "exact_norm": int(norm_equal),
                        "cer_norm": cer(pred_norm, label_norm),
                    }
                )

            if ((i // args.batch_size) + 1) % 20 == 0 or i == 0:
                print(f"[INFO] processed {min(i + args.batch_size, len(rows))}/{len(rows)}")

    total = max(1, len(results))
    print("[RESULT]")
    print(f"total={len(results)}")
    print(f"exact_match_raw={exact_raw / total:.6f}")
    print(f"exact_match_normalized={exact_norm / total:.6f}")
    print(f"mean_cer_normalized={sum(cers) / total:.6f}")

    print("[SAMPLES]")
    for row in results[:10]:
        print(f"{row['image_name']}\tlabel={row['label']}\tpred={row['pred']}")

    if args.out_csv:
        out_csv = os.path.abspath(args.out_csv)
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[SAVED] {out_csv}")


if __name__ == "__main__":
    main()
