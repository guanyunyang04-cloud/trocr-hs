import argparse
import os
from PIL import Image
import pandas as pd
import torch
import numpy as np
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutput
import hotwheels.amct_pytorch as amct

from calibration import EncoderWrapper, DecoderWrapper
from deploy.utils import convert_to_delpy


class DecoderWrapperOutput(torch.nn.Module):
    def __init__(self, decode):
        super().__init__()
        self.decode = decode

    def forward(self,
                input_ids=None,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                head_mask=None,
                cross_attn_head_mask=None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None, ):
        logits = self.decode(input_ids, attention_mask, encoder_hidden_states)
        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            input_ids = input_ids[:, -1:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }


class EncoderWrapperOutput(torch.nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values, return_dict=True):
        hidden_states = self.encoder(pixel_values)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
        )


def calculate_predicted_accuracy(names, actual_list, predicted_list):
    list_accuracy = []
    cor = 0
    for name, actual_plate, predict_plate in zip(names, actual_list, predicted_list):
        accuracy = 0
        num_matches = 0

        if actual_plate == predict_plate:
            accuracy = 1.0
            cor += 1
        else:
            for a, p in zip(actual_plate, predict_plate):
                if a == p:
                    num_matches += 1
            accuracy = round((num_matches / len(actual_plate)), 2)
        list_accuracy.append(accuracy)
    list_accuracy = np.asarray(list_accuracy)
    avg_value = np.average(list_accuracy)
    print(f'cer:{avg_value}')
    print(f'cor:{cor / len(actual_list)}')
    return avg_value


def replace_img_quant_model(model, quant_model_path):
    encoder = EncoderWrapper(model)
    encoder_qmodel = amct.restore_quant_model_fx(
        f'{quant_model_path}/trocr_encoder/config.json',
        encoder,
        f'{quant_model_path}/trocr_encoder/encoder.pt')
    encoder_qmodel = EncoderWrapperOutput(encoder_qmodel)
    setattr(encoder_qmodel, 'main_input_name', model.encoder.main_input_name)
    setattr(encoder_qmodel, 'config', model.encoder.config)
    model.encoder = encoder_qmodel


def replace_text_quant_model(model, quant_model_path):
    decoder = DecoderWrapper(model)
    decoder_qmodel = amct.restore_quant_model_fx(
        f'{quant_model_path}/trocr_decoder/config.json',
        decoder,
        f'{quant_model_path}/trocr_decoder/decoder.pt')
    decoder_qmodel = DecoderWrapperOutput(decoder_qmodel)
    setattr(decoder_qmodel, 'main_input_name', model.encoder.main_input_name)
    setattr(decoder_qmodel, 'config', model.encoder.config)
    model.decoder = decoder_qmodel


def generated_text(model, processor, arg):

    device = arg.device
    labels_csv = pd.read_csv(arg.labels_csv, sep=',')

    list_license_plates = [i for i in labels_csv[labels_csv.columns[1]]]
    list_file_names = labels_csv[labels_csv.columns[0]]
    print(list_license_plates)
    print(list_file_names)
    generated_plate_numbers = []
    for file, target in zip(list_file_names, list_license_plates):
        path = os.path.join(arg.test_dir, file)
        image = Image.open(path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f'pred:{generated_text:<15} target:{target:<15} {generated_text == target}')
        generated_plate_numbers.append(generated_text)
    return list_file_names, list_license_plates, generated_plate_numbers


def quant_inference(args):
    model_path = args.model_path
    quant_save_path = args.quant_save_path
    model = VisionEncoderDecoderModel.from_pretrained(model_path).eval()
    processor = TrOCRProcessor.from_pretrained(model_path)

    # 自动检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # 加载模型到GPU
    model = model.to(device)  # 替换为你的模型加载代码
    model = convert_to_delpy(model)

    replace_img_quant_model(model, quant_save_path)
    replace_text_quant_model(model, quant_save_path)

    amct.enable_quantization(model.encoder, fake_quant=True)
    list_file_names, list_license_plates, generated_plate_numbers = generated_text(model, processor, args)
    calculate_predicted_accuracy(list_file_names, list_license_plates, generated_plate_numbers)

    amct.enable_quantization(model.encoder, real_quant=True)
    list_file_names, list_license_plates, generated_plate_numbers = generated_text(model, processor, args)
    calculate_predicted_accuracy(list_file_names, list_license_plates, generated_plate_numbers)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./trocr-small-printed', help='the path of trocr project')
    parser.add_argument('--quant_save_path', type=str, default='/output/ptq/', help='the path of trocr quant save path')
    parser.add_argument('--test_dir', type=str, required=True, default='', help='test img path')
    parser.add_argument('--labels_csv', type=str, required=True, default='', help='label json file')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args

if __name__ == '__main__':
    args = parse_opt()
    quant_inference(args)
