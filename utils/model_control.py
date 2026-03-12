import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers.models.embeddings import apply_rotary_emb

def register_transformer_blocks_kontext(pipe):

    def get_attr(obj, attr_path: str):
        cur = obj
        for p in attr_path.split("."):
            cur = getattr(cur, p)
        return cur

    def set_attr_raw(obj, attr_path: str, value):
        parts = attr_path.split(".")
        cur = obj
        for p in parts[:-1]:
            cur = getattr(cur, p)
        setattr(cur, parts[-1], value)

    class SingleTransformerBlock(nn.Module):
        def __init__(self, orig: nn.Module, name: str):
            super().__init__()
            self.orig = orig
            self.name = name

        def forward(self, *args, **kwargs):
            return self.orig(*args, **kwargs)

    class TransformerBlock(nn.Module):
        def __init__(self, orig: nn.Module, name: str):
            super().__init__()
            self.orig = orig
            self.name = name

        def forward(self, *args, **kwargs):
            return self.orig(*args, **kwargs)

    weight_keys = pipe.transformer.state_dict().keys()
    transformer_modules = []
    single_transformer_modules = []

    for weight_key in weight_keys:
        module_name = ".".join(weight_key.split(".")[:2])  # e.g. single_transformer_blocks.7
        if weight_key.startswith("single_transformer_blocks"):
            if module_name not in single_transformer_modules:
                single_transformer_modules.append(module_name)
        elif weight_key.startswith("transformer_blocks"):
            if module_name not in transformer_modules:
                transformer_modules.append(module_name)

    for m in single_transformer_modules:
        orig = get_attr(pipe.transformer, m)
        set_attr_raw(pipe.transformer, m, SingleTransformerBlock(orig, m))

    for m in transformer_modules:
        orig = get_attr(pipe.transformer, m)
        set_attr_raw(pipe.transformer, m, TransformerBlock(orig, m))

    print(f"[register] single_transformer_blocks: {len(single_transformer_modules)}")
    print(f"[register] transformer_blocks: {len(transformer_modules)}")


def setup_kontext_pipe(pipe):

    def register_transformer_blocks_kontext(pipe):

        def get_attr(obj, attr_path: str):
            cur = obj
            for p in attr_path.split("."):
                cur = getattr(cur, p)
            return cur

        def set_attr_raw(obj, attr_path: str, value):
            parts = attr_path.split(".")
            cur = obj
            for p in parts[:-1]:
                cur = getattr(cur, p)
            setattr(cur, parts[-1], value)

        class SingleTransformerBlock(nn.Module):
            def __init__(self, orig: nn.Module, name: str):
                super().__init__()
                self.orig = orig
                self.name = name

            def forward(self, *args, **kwargs):
                return self.orig(*args, **kwargs)

        class TransformerBlock(nn.Module):
            def __init__(self, orig: nn.Module, name: str):
                super().__init__()
                self.orig = orig
                self.name = name

            def forward(self, *args, **kwargs):
                return self.orig(*args, **kwargs)

        weight_keys = pipe.transformer.state_dict().keys()
        transformer_modules = []
        single_transformer_modules = []

        for weight_key in weight_keys:
            module_name = ".".join(weight_key.split(".")[:2])  # e.g. single_transformer_blocks.7
            if weight_key.startswith("single_transformer_blocks"):
                if module_name not in single_transformer_modules:
                    single_transformer_modules.append(module_name)
            elif weight_key.startswith("transformer_blocks"):
                if module_name not in transformer_modules:
                    transformer_modules.append(module_name)

        for m in single_transformer_modules:
            orig = get_attr(pipe.transformer, m)
            set_attr_raw(pipe.transformer, m, SingleTransformerBlock(orig, m))

        for m in transformer_modules:
            orig = get_attr(pipe.transformer, m)
            set_attr_raw(pipe.transformer, m, TransformerBlock(orig, m))
        print(f"[register] single_transformer_blocks: {len(single_transformer_modules)}")
        print(f"[register] transformer_blocks: {len(transformer_modules)}")



    register_transformer_blocks_kontext(pipe)
