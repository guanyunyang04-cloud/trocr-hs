import types

import torch
import picovision.nn as pico_nn
from .modules import apply_module_patches


def convert_to_delpy(model):
    model.eval()
    model.float()
    print("Convert the model to an equivalent inference implementation")
    apply_module_patches()
    replace_layernorm_with_custom(model)

    return model.eval()


def replace_layernorm_with_custom(model, prefix=""):
    for name, child in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.LayerNorm):
            print(f"Replacing pico_nn.LayerNorm at: {full_name} ({child.__class__.__name__})")
            # print(f"    └── normalized_shape: {child.normalized_shape}, eps: {child.eps}")

            custom_ln = pico_nn.LayerNorm(
                normalized_shape=child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
                bias=child.bias,
                axis=-1,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            if child.elementwise_affine:
                with torch.no_grad():
                    custom_ln.weight.copy_(child.weight)
                    custom_ln.bias.copy_(child.bias)
            setattr(model, name, custom_ln)
        else:
            replace_layernorm_with_custom(child, prefix=full_name)

import torch, inspect

'''def replace_layernorm_with_custom(module, prefix=""):
    for name, child in module.named_children():
        full = f"{prefix}.{name}" if prefix else name

        if isinstance(child, torch.nn.LayerNorm):
            # 归一化维度：有些实现接受 int
            norm_shape = child.normalized_shape
            if isinstance(norm_shape, (tuple, list)):
                norm_shape = norm_shape[-1] if len(norm_shape) > 1 else norm_shape[0]

            # 构造自定义 LN（注意：不传 bias）
            custom_ln = pico_nn.LayerNorm(
                normalized_shape=norm_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
                axis=-1,  # 你原来就是 -1，保持一致；pico 默认是 -3
                device=(child.weight.device if getattr(child, "weight", None) is not None else None),
                dtype=(child.weight.dtype  if getattr(child, "weight", None) is not None else None),
            )

            # 迁移 & 拷贝参数（仅在 elementwise_affine=True 且目标实现具备这些参数时）
            custom_ln = custom_ln.to(child.weight.device)
            custom_ln = custom_ln.to(dtype=child.weight.dtype)
            if getattr(child, "elementwise_affine", False):
                with torch.no_grad():
                    if hasattr(custom_ln, "weight") and custom_ln.weight is not None:
                        custom_ln.weight.copy_(child.weight)
                    if hasattr(custom_ln, "bias") and custom_ln.bias is not None:
                        custom_ln.bias.copy_(child.bias)

            setattr(module, name, custom_ln)

        else:
            replace_layernorm_with_custom(child, prefix=full)'''

