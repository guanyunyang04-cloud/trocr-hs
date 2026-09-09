# TrOCR Middle East License Plate Recognition

Training, evaluation, distillation, post-training quantization, and deployment utilities for a TrOCR-based OCR pipeline targeting non-standard Middle East license plates.

## Scope

This repository contains the reusable source code and configuration used for:

- fine-tuning TrOCR on cropped license-plate images;
- evaluation with character error rate and exact-match style checks;
- encoder/decoder calibration and post-training quantization;
- model conversion and deployment-oriented wrappers;
- CTC fallback experiments for constrained inference environments.

Datasets, trained checkpoints, proprietary binaries, and generated outputs are intentionally excluded. Prepare those assets locally and pass their paths through the command-line arguments or YAML configuration files.

## Repository layout

```text
.
├── train_trocr_v2.py                 # TrOCR fine-tuning
├── eval_trocr.py                     # Evaluation
├── calibration.py                    # Encoder/decoder calibration and PTQ
├── calibration_benchmark.py          # Quantized-model evaluation
├── train_ctc_fallback.py             # CTC fallback training
├── infer_ctc_fallback*.py            # CTC fallback inference
├── trocr_filter_like_train.py        # Filtering/analysis experiment
├── configs/                           # Training configuration examples
├── qconfig/                           # Quantization configuration examples
└── deploy/                            # Deployment wrappers and utilities
```

## Data format

The training and evaluation scripts expect a CSV with two columns:

```csv
file_name,label
01V0365.jpg,01V0365
```

Images should be stored in the directory supplied to the script. Keep the dataset outside this repository when it contains customer, partner, or otherwise non-public data.

## Environment

The original experiments used Python 3.7, PyTorch, Transformers, OpenCV, NumPy, Pandas, and the AMCT/PicoVision deployment toolchain. Exact versions may depend on the target accelerator and deployment environment. The quantization scripts import `hotwheels.amct_pytorch` and `picovision`; install those proprietary packages separately when reproducing PTQ or deployment steps.

Typical open-source dependencies include:

```text
torch
torchvision
transformers
accelerate
jiwer
numpy
pandas
opencv-python
onnx
onnxsim
onnxruntime
sentencepiece
PyYAML
```

## Example commands

Fine-tuning:

```bash
python train_trocr_v2.py --cfg ./configs/trocr_samll_printed_config.yml
```

Evaluation:

```bash
python eval_trocr.py \
  --model ./trocr-small-printed \
  --csv ./dataset/val.csv \
  --image-dir ./dataset/val \
  --out-csv ./output/eval/base_val.csv
```

Calibration and post-training quantization:

```bash
python calibration.py \
  --model_path ./trocr-small-printed \
  --quant_save_path ./output/ptq \
  --calibration_image_list ./dataset/calibration_img_list.txt \
  --encoder_config_def ./qconfig/encoder_custom_config.yml \
  --decoder_config_def ./qconfig/decoder_custom_config.yml
```

Before running the examples, replace the paths with local copies of the model and dataset. Do not commit model weights, calibration images, or evaluation outputs.

## Notes

- The filename `trocr_samll_printed_config.yml` is retained for compatibility with the original experiment.
- The project is intended as a research and engineering reference rather than a turnkey production package.
- Check the license terms of the upstream TrOCR model and any accelerator-specific SDK before redistribution.
