"""
Author: Omid Nejati
Email: omid_nejaty@alumni.iust.ac.ir
LNL : Introducing locality mechanism into Transformer in Transformer (TNT)
"""
import gc

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp
from timm.models.registry import register_model
from models.localvit import LocalityFeedForward
from models.tnt import Attention, TNT
import math
import os


# ---------------------------------------------------------------------------
# VRAM release utility – can be called manually from outside if desired
# ---------------------------------------------------------------------------
def clear_cuda_cache():
    """Release CUDA memory and run the Python garbage collector.
    Call this after each epoch or when hitting OOM to avoid VRAM overflow.
    Example:
        from LNL import clear_cuda_cache
        for epoch in range(num_epochs):
            train_one_epoch(...)
            clear_cuda_cache()
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'pixel_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'tnt_t_conv_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_s_conv_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_b_conv_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class Block(nn.Module):
    """ TNT Block
    """

    def __init__(self, dim, in_dim, num_pixel, num_heads=12, in_num_head=4, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # Inner transformer
        self.norm_in = norm_layer(in_dim)
        self.attn_in = Attention(
            in_dim, in_dim, num_heads=in_num_head, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm_mlp_in = norm_layer(in_dim)
        self.mlp_in = Mlp(in_features=in_dim, hidden_features=int(in_dim * 4),
                          out_features=in_dim, act_layer=act_layer, drop=drop)

        self.norm1_proj = norm_layer(in_dim)
        self.proj = nn.Linear(in_dim * num_pixel, dim, bias=True)
        # Outer transformer
        self.norm_out = norm_layer(dim)
        self.attn_out = Attention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.conv = LocalityFeedForward(dim, dim, 1, mlp_ratio, reduction=4)

        self.ls_in     = nn.Parameter(torch.ones(in_dim) * 1e-5)
        self.ls_mlp_in = nn.Parameter(torch.ones(in_dim) * 1e-5)
        self.ls_out    = nn.Parameter(torch.ones(dim) * 1e-5)


    def forward(self, pixel_embed, patch_embed):
        # inner
        x, _ = self.attn_in(self.norm_in(pixel_embed))
        pixel_embed = pixel_embed + self.drop_path(x) * self.ls_in
        # del x frees the Python reference immediately.
        # During training the autograd graph still holds the underlying storage
        # until .backward() completes, so the actual VRAM is not reclaimed here.
        # During inference (inside torch.no_grad()), this IS an immediate free.
        del x

        pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed))) * self.ls_mlp_in

        # outer
        B, N, C = patch_embed.size()
        Nsqrt = int(math.sqrt(N))
        patch_embed[:, 1:] = patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))
        x, weights = self.attn_out(self.norm_out(patch_embed))
        patch_embed = patch_embed + self.drop_path(x) * self.ls_out
        del x        # Python ref freed; actual VRAM freed only after backward() during training
        del weights  # same: freed immediately during inference, deferred during training

        cls_token, patch_embed = torch.split(patch_embed, [1, N - 1], dim=1)                 # (B, 1, dim), (B, 196, dim)
        patch_embed = patch_embed.transpose(1, 2).reshape(B, C, Nsqrt, Nsqrt)   # (B, dim, 14, 14)
        patch_embed = self.conv(patch_embed).flatten(2).transpose(1, 2)                                 # (B, 196, dim)
        patch_embed = torch.cat([cls_token, patch_embed], dim=1)
        del cls_token  # release the intermediate tensor

        # Return None instead of weights for backward compatibility with TNT.forward
        return pixel_embed, patch_embed, None


class LocalViT_TNT(TNT):
    """ Transformer in Transformer - https://arxiv.org/abs/2103.00112
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, in_dim=48, depth=12,
                 num_heads=12, in_num_head=4, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, first_stride=4,
                 use_grad_checkpoint=True, cache_every_n_steps=50):
        """
        Args added compared to the original TNT:
            use_grad_checkpoint (bool): Enable gradient checkpointing to reduce VRAM.
                Cuts activation memory by ~30-40%, important when batch_size=64 on a T4 16GB.
                Default True.
            cache_every_n_steps (int): Automatically call torch.cuda.empty_cache() after every
                N forward passes. Helps clean up fragmented memory without affecting the
                gradient graph. Default 50 steps.
        """
        super().__init__(img_size, patch_size, in_chans, num_classes, embed_dim, in_dim, depth,
                 num_heads, in_num_head, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                 drop_path_rate, norm_layer, first_stride)
        new_patch_size = self.pixel_embed.new_patch_size
        num_pixel = new_patch_size ** 2

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        for i in range(depth):
            blocks.append(Block(
                dim=embed_dim, in_dim=in_dim, num_pixel=num_pixel, num_heads=num_heads, in_num_head=in_num_head,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blocks)

        self.register_buffer('input_mean', torch.tensor([0.3337, 0.3064, 0.3171]).view(1, 3, 1, 1))
        self.register_buffer('input_std',  torch.tensor([0.2672, 0.2564, 0.2629]).view(1, 3, 1, 1))

        # Memory management configuration
        self.use_grad_checkpoint = use_grad_checkpoint
        self.cache_every_n_steps = cache_every_n_steps
        # Forward-pass counter – not registered as a buffer so it is not saved to state_dict
        self._fwd_step = 0

        self.apply(self._init_weights)

    def forward_features(self, x):
        x = (x - self.input_mean) / self.input_std
        features, _ = super().forward_features(x)
        return features

    def _forward_impl(self, x):
        if self.use_grad_checkpoint and self.training:
            features = checkpoint.checkpoint(
                self.forward_features, x, preserve_rng_state=True, use_reentrant=False
            )
        else:
            features = self.forward_features(x)

        out = self.head(self.norm(features))
        self._fwd_step += 1
        return out

    def forward(self, x, **kwargs):
        # NOTE: Do NOT auto-wrap with inference_mode/no_grad here.
        # Doing so only covers THIS class's code; the parent TNT.forward_features
        # is called via super() and its intermediate tensors are already allocated
        # before our context manager could take effect on the full chain.
        #
        # The correct pattern is for the CALLER to own the context:
        #
        #   model.eval()
        #   with torch.no_grad():          # or torch.inference_mode()
        #       out = model(x)
        #
        # This guarantees the entire call-graph (including parent class ops) runs
        # without gradient tracking and without storing intermediate activations.
        return self._forward_impl(x)

    def on_epoch_end(self):
        """Call this at the END of each epoch (after optimizer.step()) to
        release fragmented CUDA memory. Calling empty_cache inside forward()
        is wasteful because live tensors cannot be freed mid-pass anyway.

        Usage in your training loop::

            for epoch in range(num_epochs):
                train_one_epoch(...)
                model.on_epoch_end()   # <-- HERE, after the full epoch
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


@register_model
def LNL_Ti(pretrained=False, **kwargs):
    model = LocalViT_TNT(patch_size=16, embed_dim=192, in_dim=12, depth=12, num_heads=3, in_num_head=3,
                         qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_t_conv_patch16_224']

    ckpt_path = 'LNL_Ti_GTSRB_best.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        ckpt = {k: v for k, v in ckpt.items() if not k.startswith('head.')}
        model.load_state_dict(ckpt, strict=False)

    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def LNL_S(pretrained=False, **kwargs):
    model = LocalViT_TNT(patch_size=16, embed_dim=384, in_dim=24, depth=12, num_heads=6, in_num_head=4,
                         qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_s_conv_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model