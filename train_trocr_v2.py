import argparse
import glob
import inspect
import os
import shutil
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import yaml
from PIL import Image
import torch
import torch.utils.data
from PIL import ImageDraw, ImageFont
from torchvision.transforms import transforms
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator, EarlyStoppingCallback
)
from jiwer import cer
import warnings

warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")


def safe_import_convert_to_delpy():
    """Use deploy-time model patching when available; otherwise keep training path portable."""
    try:
        from deploy.utils import convert_to_delpy  # type: ignore
        return convert_to_delpy
    except Exception as exc:
        print(f"[WARN] deploy patch is unavailable, skip convert_to_delpy: {exc}")

        def _noop(model):
            return model

        return _noop


def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Cfg:
    def __init__(self, d=None):
        if d:
            for k, v in d.items():
                setattr(self, k, self._wrap(v))

    def _wrap(self, val):
        if isinstance(val, dict):
            return Cfg(val)
        elif isinstance(val, list):
            return [self._wrap(i) for i in val]
        return val

    def __setattr__(self, key, value):
        super().__setattr__(key, self._wrap(value))  # 添加属性时自动转换


class LPDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=28, transform=None):  # 9
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx].strip()
        # prepare image (i.e. resize + normalize)

        # img = cv2.imread(self.root_dir + file_name)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                      cv2.THRESH_BINARY_INV, 11, 2)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        # cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # denoised = cv2.medianBlur(gray, 3)
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # sharpened = cv2.filter2D(img, -1, kernel)
        # norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        # cv2.imwrite('temp.png', img)
        # image = Image.open('temp.png').convert("RGB")
        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length,
                                          return_tensors="pt"
                                          ).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        # labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        # # print(labels)
        # encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoding = {"pixel_values": pixel_values.squeeze(0), "labels": labels.squeeze(0)}
        return encoding


class CustomOCRDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, processor, csv_file, max_target_length=20, transform=None):
        self.root_dir = os.path.abspath(root_dir)
        self.df = pd.read_csv(csv_file, sep=',')
        self.img_paths = self.df['file_name']
        self.labels = self.df['label']
        self.processor = processor
        self.transform = transform
        self.max_target_length = max_target_length


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # The image file name.
        file = os.path.join(self.root_dir, self.img_paths[idx])
        text = self.labels[idx]
        # Read the image, apply augmentations, and get the transformed pixels.
        image = Image.open(file).convert('RGB')
        if self.transform:
            image = self.transform(image)
        pixel_values = self.processor(image, return_tensors='pt').pixel_values
        # Pass the text through the tokenizer and get the labels,
        labels = self.processor(
            text=text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt"
        ).input_ids
        # We are using -100 as the padding token.
        # labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        # labels = torch.tensor([label.item() if label != self.processor.tokenizer.pad_token_id else -100 for label in labels.squeeze(0)])
        # encoding = {"pixel_values": pixel_values.squeeze(0), "labels": labels}
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoding = {"pixel_values": pixel_values.squeeze(0), "labels": labels.squeeze(0)}
        # encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        # print(encoding)
        # return pixel_values.squeeze(0), torch.tensor(labels).squeeze(0)
        return encoding


def get_lpd_datasets(processor, train_transforms, dataset_cfg):
    train_df = pd.read_csv(os.path.join(dataset_cfg.dataset_folder, 'retrain-labels.csv'), sep=',')
    test_df = pd.read_csv(os.path.join(dataset_cfg.dataset_folder, 'DataTestLabelled.csv'), sep=',')

    train_df = train_df.rename(columns={train_df.columns[0]: "file_name", train_df.columns[1]: "text"}, inplace=False)
    test_df = test_df.rename(columns={test_df.columns[1]: "text", test_df.columns[0]: "file_name"}, inplace=False)

    train_dataset = LPDataset(root_dir=os.path.join(os.getcwd(), dataset_cfg.train_dir),
                              df=train_df,
                              processor=processor, transform=train_transforms)
    eval_dataset = LPDataset(root_dir=os.path.join(os.getcwd(), dataset_cfg.val_dir),
                             df=test_df,
                             processor=processor)
    return train_dataset, eval_dataset


def init_special_token(model, processor):
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id


def init_transformers():
    train_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=8, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            # transforms.RandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
        ], p=0.3),
        # transforms.RandomApply([
        #     transforms.RandomErasing(scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
        # ], p=0.3),
    ])
    return train_transforms


def load_draw_font(size=20):
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/DejaVuSans.ttf",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def summarize_vocab(tokenizer, preview_size=20):
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
    preview = [repr(token) for token, _ in vocab[:preview_size]]
    return {
        "size": len(vocab),
        "preview": preview,
        "bos_token": repr(tokenizer.bos_token),
        "eos_token": repr(tokenizer.eos_token),
        "pad_token": repr(tokenizer.pad_token),
        "unk_token": repr(tokenizer.unk_token),
    }


def main(args):
    seed_everything(42)
    cfg = args.cfg
    model_cfg = cfg.model_config
    dataset_cfg = cfg.dataset_config
    train_cfg = cfg.train_config
    # ---------- 设置参数 ----------
    output_dir = f"./output/train/{train_cfg.output_dir_prefix}"
    cfg.train_config.output_dir = output_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.train_config.device = device
    # ---------- 加载模型与处理器 ----------
    processor = TrOCRProcessor.from_pretrained(model_cfg.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_cfg.model_name, ignore_mismatched_sizes=True)
    convert_to_delpy = safe_import_convert_to_delpy()
    model = convert_to_delpy(model)
    model.to(device)
    print(f'generation_config:{model.generation_config.__dict__}')

    print(f'vocab_summary:{summarize_vocab(processor.tokenizer)}')
    # for name, param in model.encoder.named_parameters():
    #     if any(name.startswith(f"encoder.layer.{i}") for i in [0, 1, 2]):
    #         param.requires_grad = False
    # train_transforms = transforms.Compose([
    #     transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
    #     transforms.ColorJitter(0.2, 0.2),
    #     transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    # ])
    train_transforms = init_transformers()

    # Model Configurations
    init_special_token(model, processor)
    # ---------- 构建数据集 ----------
    if dataset_cfg.use_lpd:
        train_dataset, val_dataset = get_lpd_datasets(processor, train_transforms, dataset_cfg)
    else:
        train_csv = os.path.join(dataset_cfg.dataset_folder, 'train.csv')
        val_csv = os.path.join(dataset_cfg.dataset_folder, 'val.csv')
        train_dataset = CustomOCRDataset(dataset_cfg.train_dir, processor, train_csv, transform=train_transforms)
        val_dataset = CustomOCRDataset(dataset_cfg.val_dir, processor, val_csv)

    # visualize_images(train_dataset, n=6)

    # print('Freeze the encoder parameters to train only the decoder')
    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    # print('Freeze the encoder parameters to train only the decoder')
    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    # Print the model to verify which parameters are frozen
    # print(model)

    def compute_cer(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

        # 将 -100 替换为 pad_token_id 以便 decode
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        # 过滤掉空字符串对（常见于 padding）
        filtered_preds = []
        filtered_labels = []

        for p, l in zip(pred_str, label_str):
            if l.strip() != "":
                filtered_preds.append(p)
                filtered_labels.append(l)

        if len(filtered_labels) == 0:
            return {"cer": None}

        # for pred, label in zip(pred_str[:10], label_str[:10]):
        #     print(f"Pred: {pred}, Label: {label}, {pred == label}")

        cer_score = cer(filtered_labels, filtered_preds)
        return {"cer": cer_score}

    callbacks = []
    if train_cfg.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=train_cfg.early_stopping_patience))

    # ---------- 训练参数 ----------
    training_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg.train_batch_size,
        per_device_eval_batch_size=train_cfg.val_batch_size,
        learning_rate=train_cfg.lr,
        num_train_epochs=train_cfg.epoch,
        save_strategy="epoch",
        warmup_ratio=0.1,
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=train_cfg.save_total_limit,
        predict_with_generate=True,
        fp16=False,
        report_to="tensorboard",
        load_best_model_at_end=train_cfg.load_best_model_at_end,
        metric_for_best_model="eval_cer",
        weight_decay=0.05,
    )
    seq2seq_args_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "evaluation_strategy" in seq2seq_args_params:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_kwargs["eval_strategy"] = "epoch"
    if "greater_is_better" in seq2seq_args_params:
        training_kwargs["greater_is_better"] = False
    if "dataloader_num_workers" in seq2seq_args_params:
        training_kwargs["dataloader_num_workers"] = getattr(train_cfg, "dataloader_num_workers", 0)
    if "fp16" in seq2seq_args_params:
        training_kwargs["fp16"] = bool(getattr(train_cfg, "fp16", False)) and device.type == "cuda"
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    # ---------- 初始化 Trainer ----------
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_cer,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )
    seq2seq_trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "tokenizer" in seq2seq_trainer_params:
        trainer_kwargs["tokenizer"] = processor.tokenizer
    elif "processing_class" in seq2seq_trainer_params:
        trainer_kwargs["processing_class"] = processor
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    # ---------- 开始训练 ----------
    trainer.train()

    # ---------- 保存模型 ----------
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # ---------- 推理测试 ----------
    model.eval()
    if dataset_cfg.use_lpd:
        eval_lpr_accuracy_and_draw_pic(model, processor, cfg)
    else:
        eval_accuracy_and_draw_pic(model, processor, cfg)


def eval_accuracy_and_draw_pic(model, processor, cfg):
    print("\n 推理测试：")
    dataset_cfg = cfg.dataset_config
    train_cfg = cfg.train_config
    pred_save_path = os.path.join(train_cfg.output_dir, 'preds')
    shutil.rmtree(pred_save_path, ignore_errors=True)
    os.makedirs(pred_save_path, exist_ok=True)
    corr = 0
    font = load_draw_font()
    img_paths = sorted(glob.glob(f'{dataset_cfg.train_dir}/**/*.jpg', recursive=True)
                       + glob.glob(f'{dataset_cfg.train_dir}/**/*.png', recursive=True)
                       + glob.glob(f'{dataset_cfg.val_dir}/**/*.jpg', recursive=True)
                       + glob.glob(f'{dataset_cfg.val_dir}/**/*.png', recursive=True)
                       )
    label_map = {}
    for csv_name in ("train.csv", "val.csv"):
        csv_path = os.path.join(dataset_cfg.dataset_folder, csv_name)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, sep=",")
        if "file_name" in df.columns and "label" in df.columns:
            for _, row in df.iterrows():
                label_map[str(row["file_name"]).strip()] = str(row["label"]).strip()

    count = len(img_paths)
    for image_path in img_paths:
        test_img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(test_img)
        pixel_values = processor(images=test_img, return_tensors="pt").pixel_values.to(train_cfg.device)
        generated_ids = model.generate(pixel_values)
        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        draw.text((0, 0), pred_text, fill=(255, 0, 0), font=font)
        test_img.save(f'{pred_save_path}/{os.path.basename(image_path)}')
        image_name = os.path.basename(image_path)
        target = label_map.get(image_name)
        if target is None:
            target = os.path.splitext(image_name)[0].split('_')[0]
            print(f"[WARN] label not found in csv for {image_name}, fallback to filename prefix: {target}")
        if pred_text.strip().upper() == target.upper():
            corr += 1
        print(f"pred: {pred_text:<15}; label: {target}; equal: {pred_text == target}")
    print(f'acc={corr / count}')


def eval_lpr_accuracy_and_draw_pic(model, processor, cfg):
    print("\n 推理测试：")
    dataset_cfg = cfg.dataset_config
    train_cfg = cfg.train_config
    pred_save_path = os.path.join(train_cfg.output_dir, 'preds')
    shutil.rmtree(pred_save_path, ignore_errors=True)
    os.makedirs(pred_save_path, exist_ok=True)
    corr = 0
    font = load_draw_font()
    test_df = pd.read_csv(os.path.join(dataset_cfg.dataset_folder, 'DataTestLabelled.csv'), sep=',')
    test_df = test_df.rename(columns={test_df.columns[1]: "text", test_df.columns[0]: "file_name"}, inplace=False)
    file_names = test_df['file_name']
    text = test_df['text']

    count = len(file_names)
    for image_path, target in zip(file_names, text):
        image_path = os.path.join(dataset_cfg.val_dir, image_path)
        test_img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(test_img)
        pixel_values = processor(images=test_img, return_tensors="pt").pixel_values.to(train_cfg.device)
        generated_ids = model.generate(pixel_values)
        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        draw.text((0, 0), pred_text, fill=(255, 0, 0), font=font)
        test_img.save(f'{pred_save_path}/{os.path.basename(image_path)}')
        if pred_text.strip().upper() == target.strip().replace(' ', '').upper():
            corr += 1
        print(f"pred: {pred_text:<15}; label: {target}; equal: {pred_text == target}")
    print(f'acc={corr / count}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='./configs/trocr_samll_printed_config.yml',
                        help='hyper train params path')

    args = parser.parse_args()
    cfg_file = args.cfg
    with open(cfg_file, 'r') as f:
        cfg = Cfg(yaml.safe_load(f))
    cfg.dataset_config.train_dir = os.path.join(cfg.dataset_config.dataset_folder, cfg.dataset_config.train_dir)
    cfg.dataset_config.val_dir = os.path.join(cfg.dataset_config.dataset_folder, cfg.dataset_config.val_dir)
    args.cfg = cfg
    return args


if __name__ == '__main__':
    args = parse_opt()
    main(args)
