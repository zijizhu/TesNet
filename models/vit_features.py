import os
import re
from functools import partial
from math import sqrt

import numpy as np
import torch
from dinov2.layers.block import Block, MemEffAttention
from dinov2.models.vision_transformer import DinoVisionTransformer as Dinov2VisionTransformer
from .dino_backbone.vision_transformer import VisionTransformer
from einops import rearrange
from torch import nn
from logging import getLogger

from .maskclip import clip

logger = getLogger(__name__)

def block_expansion_dino(state_dict: dict[str, torch.Tensor], n_splits: int = 3):
    """Perform Block Expansion on a ViT described in https://arxiv.org/abs/2404.17245"""
    block_keys = set(re.search("^blocks.(\d+).", key).group(0) for key in state_dict if key.startswith("blocks."))
    n_blocks = len(block_keys)
    
    block_indices = np.arange(0, n_blocks).reshape((n_splits, -1,))
    block_indices = np.concatenate([block_indices, block_indices[:, -1:]], axis=-1)
    
    n_splits, n_block_per_split = block_indices.shape
    new_block_indices = list((i + 1) * n_block_per_split - 1 for i in range(n_splits))
    
    expanded_state_dict = dict()
    learnable_param_names, zero_param_names = [], []
    
    for dst_idx, src_idx in enumerate(block_indices.flatten()):
        src_keys = [k for k in state_dict if f"blocks.{src_idx}" in k]
        dst_keys = [k.replace(f"blocks.{src_idx}", f"blocks.{dst_idx}") for k in src_keys]
        
        block_state_dict = dict()
        
        for src_k, dst_k in zip(src_keys, dst_keys):
            if ("mlp.fc2" in dst_k or "attn.proj" in dst_k) and (dst_idx in new_block_indices):
                block_state_dict[dst_k] = torch.zeros_like(state_dict[src_k])
                zero_param_names.append(dst_k)
            else:
                block_state_dict[dst_k] = state_dict[src_k]

        expanded_state_dict.update(block_state_dict)

        if dst_idx in new_block_indices:
            learnable_param_names += dst_keys

    expanded_state_dict.update({k: v for k, v in state_dict.items() if "block" not in k})
    
    return expanded_state_dict, len(block_indices.flatten()), learnable_param_names, zero_param_names


dinov2_common_kwargs = dict(
    img_size=518,
    patch_size=14,
    mlp_ratio=4,
    init_values=1.0,
    ffn_layer="mlp",
    block_chunks=0,
    num_register_tokens=4,
    interpolate_antialias=True,
    interpolate_offset=0.0,
    block_fn=partial(Block, attn_class=MemEffAttention)
)

dino_common_kwargs = dict(
    num_classes=0,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

vit_small_kwargs = dict(embed_dim=384, num_heads=6)
vit_base_kwargs = dict(embed_dim=768, num_heads=12)

MODEL_DICT = {
    "dinov2_vits14_reg4": partial(Dinov2VisionTransformer, **vit_small_kwargs, **dinov2_common_kwargs),
    "dinov2_vitb14_reg4": partial(Dinov2VisionTransformer, **vit_base_kwargs, **dinov2_common_kwargs),
    "dino_vits16": partial(VisionTransformer, patch_size=16, **vit_small_kwargs, **dino_common_kwargs),
    "dino_vits8": partial(VisionTransformer, patch_size=8, **vit_small_kwargs, **dino_common_kwargs),
    "dino_vitb16": partial(VisionTransformer, patch_size=16, **vit_base_kwargs, **dino_common_kwargs),
    "dino_vitb8": partial(VisionTransformer, patch_size=8, **vit_base_kwargs, **dino_common_kwargs)
}
URL_DICT = {
    "dinov2_vits14_reg4": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14_reg4": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
    "dino_vits16": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
    "dino_vits8": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
    "dino_vitb16": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
    "dino_vitb8": "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
}

DIM_DICT = {
    "dinov2_vits14_reg4": 384,
    "dinov2_vitb14_reg4": 768,
    "dino_vits16": 384,
    "dino_vits8": 384,
    "dino_vitb16": 768,
    "dino_vitb8": 768,
}


class DINOv2Backbone(nn.Module):
    def __init__(self, name: str = "dinov2_vitb14_reg", pretrained=True):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", name)
        self.dim = DIM_DICT[name]

    def forward(self, x: torch.Tensor, key: str = "x_norm_patchtokens", reshape: bool = False) -> torch.Tensor:
        feature_dict = self.dino.forward_features(x)  # type: dict[str, torch.Tensor]
        feature = feature_dict[key]
        
        B, n_patches, dim = feature.shape

        if reshape and key == "x_norm_patch_tokens":
            H = W = int(sqrt(n_patches))
            feature = rearrange(feature, "B (H W) dim -> B dim H W", H=H, W=W)
        
        return feature


class DINOv2BackboneExpanded(nn.Module):
    def __init__(self, name: str = "dinov2_vitb14_reg4", n_splits: int = 3, pretrained=True):
        super().__init__()
        self.name = name
        self.dim = DIM_DICT[name]
        self.n_splits = n_splits
        if n_splits > 0:
            arch = MODEL_DICT[name]
            state_dict = torch.hub.load_state_dict_from_url(URL_DICT[name], map_location="cpu")
            expanded_state_dict, n_blocks, learnable_param_names, zero_param_names = block_expansion_dino(
                state_dict=state_dict,
                n_splits=n_splits)
            self.dino = arch(depth=n_blocks)
            self.dino.load_state_dict(expanded_state_dict)
            self.learnable_param_names = learnable_param_names
        else:
            self.dino = torch.hub.load('facebookresearch/dinov2', name[:-1])  # type: nn.Module
            self.learnable_param_names = []
    
    def learnable_parameters(self):
        logger.info("Setting backbone learnable params")
        return list(param for name, param in self.dino.named_parameters() if name in self.learnable_param_names)
    
    def set_requires_grad(self):
        for name, param in self.dino.named_parameters():
            param.requires_grad = name in self.learnable_param_names

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_dict = self.dino.forward_features(x)  # type: dict[str, torch.Tensor]
        feature = feature_dict["x_norm_patchtokens"]
        
        B, n_patches, dim = feature.shape

        H = W = int(sqrt(n_patches))
        feature = rearrange(feature, "B (H W) dim -> B dim H W", H=H, W=W)
        
        return feature
    
    def forward_all(self, x: torch.Tensor):
        x = self.dino.prepare_tokens_with_masks(x, masks=None)

        features = []
        for b, blk in enumerate(self.dino.blocks):
            x = blk(x)
            if b in [4, 14]:
                features.append(rearrange(x[:, self.dino.num_register_tokens + 1:, :], "B (H W) dim -> B dim H W", H=16, W=16))

        x_norm = self.dino.norm(x)
        return rearrange(x_norm[:, self.dino.num_register_tokens + 1:, :], "B (H W) dim -> B dim H W", H=16, W=16), features
    
    def __repr__(self):
        return self.name


class DINOBackboneExpanded(nn.Module):
    def __init__(self, name: str = "dino_vitb16", n_splits: int = 0, mode: str = "block_expansion", freeze_norm_layer: bool = True, pretrained=True):
        super().__init__()
        self.name = name
        self.dim = DIM_DICT[name]
        assert mode in ["block_expansion", "append"]

        arch = MODEL_DICT[name]
        state_dict = torch.hub.load_state_dict_from_url(URL_DICT[name], map_location="cpu")
        if n_splits > 0:
            expanded_state_dict, n_blocks, learnable_param_names, zero_param_names = block_expansion_dino(
                state_dict=state_dict,
                n_splits=n_splits
            )
            self.dino = arch(depth=n_blocks)
            self.dino.load_state_dict(expanded_state_dict)
            self.learnable_param_names = learnable_param_names
        else:
            self.dino = torch.hub.load('facebookresearch/dino:main', name)
            self.learnable_param_names = []

    def learnable_parameters(self):
        logger.info("Setting backbone learnable params")
        return list(param for name, param in self.dino.named_parameters() if name in self.learnable_param_names)
    
    def set_requires_grad(self):
        for name, param in self.dino.named_parameters():
            param.requires_grad = name in self.learnable_param_names
    
    def forward_with_original_feature(self, x):
        x = self.dino.prepare_tokens(x)
        
        original_feature = None
        for i, blk in enumerate(self.dino.blocks):
            x = blk(x)
            if i == 11:
                original_feature = self.dino.norm(x)
        x = self.dino.norm(x)
        return x[:, 1:], original_feature[:, 1:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dino.prepare_tokens(x)
        
        for i, blk in enumerate(self.dino.blocks):
            x = blk(x)
        x = self.dino.norm(x)
        x =  x[:, 1:]
        batch_size, n_patches, dim = x.shape
        H = W = int(sqrt(n_patches))
        features = rearrange(x, "B (H W) D -> B D H W", H=H, W=W)
        return features

    def forward_all(self, x: torch.Tensor):
        x = self.dino.prepare_tokens(x)

        features = []
        for b, blk in enumerate(self.dino.blocks):
            x = blk(x)
            if b in [4, 14]:
                B, n_patches, dim = x.shape
                H = W = int(sqrt(n_patches))
                features.append(rearrange(x[:, 1:], "B (H W) dim -> B dim H W", H=H, W=W))

        x_norm = self.dino.norm(x)
        B, n_patches, dim = x.shape
        H = W = int(sqrt(n_patches))
        return rearrange(x_norm[:, 1:], "B (H W) dim -> B dim H W", H=H, W=W), features

    def __repr__(self):
        return self.name


class MaskCLIP(nn.Module):
    """
    Implementation adapted from https://github.com/mhamilton723/FeatUp/tree/main/featup/featurizers
    """
    def __init__(self, name: str = "ViT-B/16", pretrained=True):
        super().__init__()
        self.model, self.preprocess = clip.load(
            name,
            download_root=os.getenv('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
        )
        self.model.eval()
        self.patch_size = self.model.visual.patch_size

    def forward(self, img):
        features = self.model.get_patch_encodings(img).to(torch.float32)
        return features
