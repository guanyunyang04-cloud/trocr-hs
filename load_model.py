from transformers import VisionEncoderDecoderModel

from calibration import EncoderWrapper, DecoderWrapper
from deploy.utils import convert_to_delpy


def trocr_samll_printed_encoder():
    model_name = 'trocr-small-printed'
    model = VisionEncoderDecoderModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
    model = convert_to_delpy(model)
    encoder_model = EncoderWrapper(model)
    return encoder_model


def trocr_samll_printed_decoder():
    model_name = 'trocr-small-printed'
    model = VisionEncoderDecoderModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
    model = convert_to_delpy(model)
    decoder_model = DecoderWrapper(model)
    return decoder_model
