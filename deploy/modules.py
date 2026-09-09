import collections.abc
import math
import random
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.fx
import torch.nn.functional as F
import torch.utils.checkpoint
import picovision.nn as pico_nn
from torch import Tensor
from transformers.models.deit.modeling_deit import *
from transformers.models.trocr.modeling_trocr import  TrOCRLearnedPositionalEmbedding, \
    TrOCRDecoder, TrOCRAttention, TrOCRDecoderLayer
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings, ViTIntermediate, ViTOutput, ViTSelfOutput

from transformers.utils import add_start_docstrings, logging, replace_return_docstrings

logger = logging.get_logger(__name__)
MASK_MIN_VALUE = -1024


@torch.fx.wrap
def get_combined_attention_mask(input_shape, inputs_embeds, past_key_values_length, attention_mask):
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )
    return combined_attention_mask


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    # mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask = torch.full((tgt_len, tgt_len), torch.tensor(MASK_MIN_VALUE, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), MASK_MIN_VALUE)


def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    # combined_attention_mask = None
    # if input_shape[-1] > 1:
    #     combined_attention_mask = _make_causal_mask(
    #         input_shape,
    #         inputs_embeds.dtype,
    #         device=inputs_embeds.device,
    #         past_key_values_length=past_key_values_length,
    #     )
    # if attention_mask is not None:
    #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    #     expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
    #     combined_attention_mask = (
    #         expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
    #     )
    combined_attention_mask = get_combined_attention_mask(input_shape, inputs_embeds, past_key_values_length,
                                                          attention_mask)

    return combined_attention_mask


@torch.fx.wrap
def do_not_trace_attn_weights_check(attn_weights, bsz, num_heads, tgt_len, src_len):
    if attn_weights.size() != (bsz * num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )


@torch.fx.wrap
def do_not_trace_attention_mask_check(attention_mask, bsz, tgt_len, src_len):
    if attention_mask.size() != (bsz, 1, tgt_len, src_len):
        raise ValueError(
            f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
        )


@torch.fx.wrap
def do_not_trace_layer_head_mask_check(layer_head_mask, num_heads):
    if layer_head_mask.size() != (num_heads,):
        raise ValueError(
            f"Head mask for a single layer should be of size {(num_heads,)}, but is"
            f" {layer_head_mask.size()}"
        )


@torch.fx.wrap
def do_not_trace_attn_output_check(attn_output, bsz, tgt_len, num_heads, head_dim):
    if attn_output.size() != (bsz * num_heads, tgt_len, head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, tgt_len, head_dim)}, but is"
            f" {attn_output.size()}"
        )


def TrOCRAttentionforward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None
    bsz, tgt_len, embed_dim = hidden_states.size()

    # get query proj
    hidden_states = hidden_states.reshape(-1, embed_dim)
    query_states = self.q_proj(hidden_states).reshape(bsz, tgt_len, -1) * self.scaling
    # get key, value proj
    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        kb, kc, kw = key_value_states.shape
        key_states = self._shape(
            self.k_proj(key_value_states.reshape(-1, kw)).reshape(kb, kc, -1), -1, bsz)
        value_states = self._shape(self.v_proj(key_value_states.reshape(-1, kw)).reshape(kb, kc, -1), -1, bsz)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states).reshape(bsz, tgt_len, -1), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states).reshape(bsz, tgt_len, -1), -1, bsz)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.matmul(query_states, key_states.transpose(1, 2))

    # if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
    #     raise ValueError(
    #         f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
    #         f" {attn_weights.size()}"
    #     )
    do_not_trace_attn_weights_check(attn_weights, bsz, self.num_heads, tgt_len, src_len)
    if attention_mask is not None:
        # if attention_mask.size() != (bsz, 1, tgt_len, src_len):
        #     raise ValueError(
        #         f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
        #     )
        do_not_trace_attention_mask_check(attention_mask, bsz, tgt_len, src_len)
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        # if layer_head_mask.size() != (self.num_heads,):
        #     raise ValueError(
        #         f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
        #         f" {layer_head_mask.size()}"
        #     )
        do_not_trace_layer_head_mask_check(layer_head_mask, self.num_heads)
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if output_attentions:
        # this operation is a bit awkward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to be reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.matmul(attn_probs, value_states)

    # if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
    #     raise ValueError(
    #         f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
    #         f" {attn_output.size()}"
    #     )
    do_not_trace_attn_output_check(attn_output, bsz, tgt_len, self.num_heads, self.head_dim)

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
    b, c, w = attn_output.shape
    attn_output = self.out_proj(attn_output.reshape(-1, w)).reshape(b, c, -1)

    return attn_output, attn_weights_reshaped, past_key_value


def TrOCRDecoderLayerforward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
):
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
        attention_mask (`torch.FloatTensor`): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        encoder_hidden_states (`torch.FloatTensor`):
            cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
        encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
            `(encoder_attention_heads,)`.
        cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
            size *(decoder_attention_heads,)*.
        past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
    """
    residual = hidden_states

    # Self Attention
    # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
    self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    # add present self-attn cache to positions 1,2 of present_key_value tuple
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        past_key_value=self_attn_past_key_value,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
    )

    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)

    # Cross-Attention Block
    cross_attn_present_key_value = None
    cross_attn_weights = None

    if encoder_hidden_states is not None:
        residual = hidden_states

        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # add cross-attn to positions 3,4 of present_key_value tuple
        present_key_value = present_key_value + cross_attn_present_key_value

    # Fully Connected
    residual = hidden_states
    b, c, w = hidden_states.shape
    hidden_states = self.activation_fn(self.fc1(hidden_states.reshape(-1, w)).reshape(b, c, -1))
    hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    b, c, w = hidden_states.shape
    hidden_states = self.fc2(hidden_states.reshape(-1, w)).reshape(b, c, -1)

    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    hidden_states = residual + hidden_states
    hidden_states = self.final_layer_norm(hidden_states)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights, cross_attn_weights)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def DeiTSelfAttentionforward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    b, c, w = hidden_states.data.shape
    mixed_query_layer = self.query(hidden_states.reshape(-1, w)).reshape(b, c, -1)

    key_layer = self.transpose_for_scores(self.key(hidden_states.reshape(-1, w)).reshape(b, c, -1))
    value_layer = self.transpose_for_scores(self.value(hidden_states.reshape(-1, w)).reshape(b, c, -1))
    query_layer = self.transpose_for_scores(mixed_query_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # Normalize the attention scores to probabilities.
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    return outputs


def DeiTSelfOutputforward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    b, c, w = hidden_states.data.shape
    hidden_states = self.dense(hidden_states.reshape(-1, w)).reshape(b, c, -1)
    hidden_states = self.dropout(hidden_states)

    return hidden_states


def DeiTIntermediateforward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    b, c, w = hidden_states.data.shape
    hidden_states = self.dense(hidden_states.reshape(-1, w)).reshape(b, c, -1)
    hidden_states = self.intermediate_act_fn(hidden_states)

    return hidden_states


def DeiTOutputforward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    b, c, w = hidden_states.data.shape
    hidden_states = self.dense(hidden_states.reshape(-1, w)).reshape(b, c, -1)
    hidden_states = self.dropout(hidden_states)

    hidden_states = hidden_states + input_tensor

    return hidden_states


def DeiTModelforward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
        Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None:
        raise ValueError("You have to specify pixel_values")

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
    expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
    # if pixel_values.dtype != expected_dtype:
    #     pixel_values = pixel_values.to(expected_dtype)

    embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

    encoder_outputs = self.encoder(
        embedding_output,
        head_mask=head_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    sequence_output = encoder_outputs[0]
    sequence_output = self.layernorm(sequence_output)
    pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

    if not return_dict:
        head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
        return head_outputs + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


@torch.fx.wrap
def do_not_trace_assert(pixel_values, image_size):
    batch_size, num_channels, height, width = pixel_values.shape
    if num_channels != num_channels:
        raise ValueError(
            "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
        )
    if height != image_size[0] or width != image_size[1]:
        raise ValueError(
            f"Input image size ({height}*{width}) doesn't match model ({image_size[0]}*{image_size[1]})."
        )


def DeiTPatchEmbeddingsforward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    # batch_size, num_channels, height, width = pixel_values.shape
    # if num_channels != self.num_channels:
    #     raise ValueError(
    #         "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
    #     )
    # if height != self.image_size[0] or width != self.image_size[1]:
    #     raise ValueError(
    #         f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
    #     )
    do_not_trace_assert(pixel_values, self.image_size)
    x = self.projection(pixel_values).flatten(2).transpose(1, 2)
    return x


@torch.fx.wrap
def do_not_trace_arange(input_ids, past_key_values_length, weight):
    bsz, seq_len = input_ids.shape[:2]
    positions = torch.arange(
        past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=weight.device
    ).expand(bsz, -1)
    return positions


def TrOCRLearnedPositionalEmbeddingforward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
    """`input_ids' shape is expected to be [bsz x seqlen]."""

    # bsz, seq_len = input_ids.shape[:2]
    # positions = torch.arange(
    #     past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
    # ).expand(bsz, -1)
    positions = do_not_trace_arange(input_ids, past_key_values_length, self.weight)
    return F.embedding(
        positions, self.weight, self.padding_idx, self.max_norm,
        self.norm_type, self.scale_grad_by_freq, self.sparse)
    # return super().forward(positions + self.offset)


class PicoGELUActivation(nn.Module):
    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = pico_nn.GELU()

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


print(f'Replacing pico_nn.GELU at {str(type(ACT2FN["gelu"]))}')
ACT2FN["gelu"] = PicoGELUActivation


DEPLOY_MODULE_MAPPINGS = {
    'encode': {
        DeiTSelfAttention: {"forward": DeiTSelfAttentionforward},
        DeiTSelfOutput: {"forward": DeiTSelfOutputforward},
        DeiTIntermediate: {"forward": DeiTIntermediateforward},
        DeiTOutput: {"forward": DeiTOutputforward},
        DeiTModel: {"forward": DeiTModelforward},
        DeiTPatchEmbeddings: {"forward": DeiTPatchEmbeddingsforward},
        # ViTLayer.layernorm_before layernorm_after  # 换成picov ln}
    },
    'decode': {
        TrOCRLearnedPositionalEmbedding: {"forward": TrOCRLearnedPositionalEmbeddingforward},
        # # TrOCRDecoder: {"forward": TrOCRDecoderforward},
        TrOCRDecoder: {"_prepare_decoder_attention_mask": _prepare_decoder_attention_mask},
        TrOCRAttention: {"forward": TrOCRAttentionforward},
        TrOCRDecoderLayer: {"forward": TrOCRDecoderLayerforward},
    },
}

def apply_module_patches():
    modes = ['encode', 'decode']
    for mode in modes:
        mapping = DEPLOY_MODULE_MAPPINGS[mode]
        for k1, v1 in mapping.items():
            for k2, v2 in v1.items():
                print(f"[Patch] {str(k1)}.{k2} has been replaced with {v2.__name__}")
                setattr(k1, k2, v2)
