# -*- coding: UTF-8 -*-
import copy
import os
import time
import onnx
import argparse
import torch
import torch.optim
import torch.utils.data
import cv2
import numpy as np
import torch.nn as nn
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from onnxsim import simplify
import hotwheels.amct_pytorch as amct
from picovision import nn as pico_nn

from deploy.utils import convert_to_delpy

BATCH_SIZE_ENCODER = 1
BATCH_SIZE_DECODER = 1
IMAGE_WIDTH = 384
IMAGE_HEIGHT = 384

MAX_LENGTH = 20
VOCAB_BOS = 0
VOCAB_EOS = 2

SCALE_IDX = 0


class EncoderWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.encoder = model.get_encoder()

    def forward(self, x):
        res = self.encoder(x, return_dict=True)
        return res.last_hidden_state


class DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.get_decoder()

    def forward(self, text_tokens, attention_mask, last_hidden_states):
        res = self.model(input_ids=text_tokens,
                         attention_mask=attention_mask,
                         encoder_hidden_states=last_hidden_states,
                         encoder_attention_mask=None,
                         inputs_embeds=None,
                         output_attentions=False,
                         output_hidden_states=False,
                         use_cache=False,
                         past_key_values=None,
                         return_dict=True, )
        return res.logits


def get_image_from_txt(label_file):
    images = []
    with open(label_file, 'r') as file_open:
        lines = file_open.readlines()
        for line in lines:
            images.append(line.strip())
    return images


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor):
        image_paths = get_image_from_txt(dataset)
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        return image


def create_dataloader(dataset, processor, batch_size):
    val_loader = torch.utils.data.DataLoader(
        ImagesDataset(dataset, processor),
        batch_size=batch_size,
        shuffle=False
    )

    return val_loader


@torch.no_grad()
def encoder_forward(encoder, val_loader, args):
    device = args.device
    start = time.perf_counter()
    for i, images in enumerate(val_loader):
        images = images.to(device)
        encoder.to(device)
        # np.save('img.npy', images.cpu().numpy())
        # exit()
        output = encoder(images)
        print('Calibrate encoder batch [%d]' % (i))

    end = time.perf_counter()
    print('Func[encoder_forward] Finish. Time Cost:[%.6f s]' % (end - start))


@torch.no_grad()
def decoder_forward(decoder, encoder, val_loader, processor, args):
    start = time.perf_counter()
    device = args.device
    for idx, images in enumerate(val_loader):
        images = images.to(device)
        encoder.to(device)
        decoder.to(device)

        last_hidden_states = encoder(images)

        text_tokens = torch.randint(4, processor.tokenizer.vocab_size - 1, (1, MAX_LENGTH), dtype=torch.int32).to(
            device)
        # start token is 0
        text_tokens[:, 0] = 0
        # text_tokens = torch.ones(BATCH_SIZE_DECODER, MAX_LENGTH, dtype=torch.int32).to(device)*2
        attention_mask = torch.ones(BATCH_SIZE_DECODER, MAX_LENGTH, dtype=torch.int32).to(device)
        # np.save('text_tokens.npy', text_tokens.cpu().numpy())
        # np.save('attention_mask.npy', text_tokens.cpu().numpy())
        # np.save('last_hidden_states.npy', last_hidden_states.cpu().numpy())
        # exit()
        logits = decoder(text_tokens, attention_mask, last_hidden_states)[0]
    end = time.perf_counter()
    print('Func[decoder_forward] Finish. Time Cost:[%.6f s]' % (end - start))


def export_torch_encoder_fx(model, processor, args):
    result_path = os.path.join(args.quant_save_path, 'trocr_encoder')
    device = args.device
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    print('****** Export Encoder ******')
    encoder_model = EncoderWrapper(model)
    float_model = copy.deepcopy(encoder_model)
    encoder_model.eval()
    val_loader = create_dataloader(args.calibration_image_list, processor, BATCH_SIZE_ENCODER)

    encoder_forward(float_model, val_loader, args)

    print('==> [AMCT]: create quant config..')
    config_file = os.path.join(result_path, 'config.json')
    amct.create_quant_config_fx(config_file, encoder_model, args.encoder_config_def)

    print('==> [AMCT]: create quant model..')
    quant_model = amct.create_quant_model_fx(config_file, encoder_model)

    print("==> [AMCT]: do calibration..")
    amct.enable_quantization(quant_model, calibration=True)

    encoder_forward(quant_model, val_loader, args)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print('==> [AMCT]: save quantized pt model..')
    torch.save(quant_model.state_dict(), os.path.join(result_path, 'encoder.pt'))

    print('==> [AMCT]: save_quant_model_fx..')
    result_file = os.path.join(result_path, 'encoder')
    random_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH).to(device)
    amct.save_quant_model_fx(quant_model, result_file, random_input)
    if args.quant_analyze:
        for i, images in enumerate(val_loader):
            images = images.to(device)
            quant_model.to(device)
            break
        analyzer = amct.QuantAnalyzer(quant_model, images, os.path.join(result_path, 'analyze'))
        analyzer.analyze()


def export_torch_decoder_fx(model, processor, args):
    result_path = os.path.join(args.quant_save_path, 'trocr_decoder')
    device = args.device
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    print('****** Export Decoder ******')
    encoder_model = EncoderWrapper(model)
    encoder_model.eval().to(device)
    global SCALE_IDX
    SCALE_IDX = 0
    decoder_model = DecoderWrapper(model)
    decoder_model.eval()
    encoder_ouput = encoder_model(torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH).to(device))
    val_loader = create_dataloader(args.calibration_image_list, processor, BATCH_SIZE_DECODER)

    print('==> [AMCT]: create quant config..')
    config_file = os.path.join(result_path, 'config.json')
    amct.create_quant_config_fx(config_file, decoder_model, args.decoder_config_def)

    print('==> [AMCT]: create quant model..')
    quant_model = amct.create_quant_model_fx(config_file, decoder_model)

    print("==> [AMCT]: do calibration..")
    amct.enable_quantization(quant_model, calibration=True)

    decoder_forward(quant_model, encoder_model, val_loader, processor, args)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print('==> [AMCT]: save quantized pt model..')
    torch.save(quant_model.state_dict(), os.path.join(result_path, 'decoder.pt'))

    print('==> [AMCT]: save_quant_model_fx..')
    result_file = os.path.join(result_path, 'decoder')
    _text_tokens = torch.ones(1, MAX_LENGTH, dtype=torch.int32).to(device)
    _attention_mask = torch.ones(1, MAX_LENGTH, dtype=torch.int32).to(device)
    _last_hidden_states = torch.randn(encoder_ouput.shape).to(device)
    _input = (_text_tokens, _attention_mask, _last_hidden_states)
    print("### text_tokens shape:", _text_tokens.shape)
    print("### attention_mask shape:", _attention_mask.shape)
    print("### last_hidden_state shape:", _last_hidden_states.shape)
    amct.save_quant_model_fx(quant_model, result_file, _input)
    if args.quant_analyze:
        for i, images in enumerate(val_loader):
            images = images.to(device)
            encoder_model.to(device)
            quant_model.to(device)

            last_hidden_states = encoder_model(images)
            torch.manual_seed(0)
            text_tokens = torch.randint(4, processor.tokenizer.vocab_size - 1, (1, MAX_LENGTH), dtype=torch.int32).to(
                device)
            # start token is 0
            text_tokens[:, 0] = 0
            attention_mask = torch.ones(BATCH_SIZE_DECODER, MAX_LENGTH, dtype=torch.int32).to(device)
            break
        analyzer = amct.QuantAnalyzer(quant_model, (text_tokens, attention_mask, last_hidden_states),
                                      os.path.join(result_path, 'analyze'))
        analyzer.analyze()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./trocr-small-printed', help='the path of trocr project')
    parser.add_argument('--quant_save_path', type=str, default='./output/ptq', help='the path of trocr quant save path')
    parser.add_argument('--calibration_image_list', type=str, default='./dataset/calibration_img_list.txt',
                        help='image list txt file')
    parser.add_argument('--encoder_config_def', type=str, default='./qconfig/encoder_custom_config.yml',
                        help='encoder custom config def yaml')
    parser.add_argument('--decoder_config_def', type=str, default='./qconfig/decoder_custom_config.yml',
                        help='encoder custom config def yaml')
    parser.add_argument('--quant_analyze', action='store_true', help='the path of trocr project')

    args = parser.parse_args()

    quant_save_path = args.quant_save_path
    if not os.path.exists(quant_save_path):
        os.makedirs(quant_save_path)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


if __name__ == "__main__":
    args = parse_opt()

    # Load processor and model
    processor = TrOCRProcessor.from_pretrained(args.model_path)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_path, ignore_mismatched_sizes=True)
    model = convert_to_delpy(model)
    print("===> load processor and model.\n")

    # Export encoder and decoder
    export_torch_encoder_fx(model, processor, args)
    export_torch_decoder_fx(model, processor, args)
