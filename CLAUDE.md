# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

Official implementation of **"Robust Transformer with Locality Inductive Bias and Feature Normalization"** (Manzari et al., 2023, *Engineering Science and Technology, an International Journal* — paper: https://arxiv.org/abs/2301.11553).

The repo introduces **LNL (Locality iN Locality)** — Vision Transformers that inject a locality inductive bias into Transformer-in-Transformer (TNT) by replacing the MLP block with a depth-wise convolutional **LocalityFeedForward**. A second variant, **LNL-MoEx**, adds Moment Exchange feature normalization for stronger augmentation.

## Stack

- **Language**: Python
- **Framework**: PyTorch
- **Core dependency**: [`timm`](https://github.com/rwightman/pytorch-image-models) (pytorch-image-models) — used for layers (`DropPath`, `Mlp`, `trunc_normal_`), data constants, model registry (`@register_model`), and pretrained weight loading
- **Auxiliary**: `ml_collections` (config dicts for ViT baselines)
- **Target task**: ImageNet-1K image classification, 224×224 input

## Repository Layout

```
.
├── LNL.py                  # Top-level model: LNL_Ti, LNL_S (locality + TNT)
├── LNL_MoEx.py             # Top-level model: LNL_MoEx_Ti, LNL_MoEx_S (locality + TNT + MoEx)
├── Instructions.ipynb      # Original training/eval procedure (data prep, commands, expected output)
├── Instructions-best.ipynb # Optimized training/eval procedure with checkpoints, caching and evaluation
├── LNL_Ti_GTSRB_best.pt    # Best trained LNL_Ti model checkpoint
├── model_checkpoint_epoch_5.pt # Trained LNL model checkpoint at epoch 5
├── model_checkpoint_LNL_MoEx_final_epoch_5.pt # Trained LNL_MoEx model checkpoint at epoch 5
├── README.md               # Paper info and citation
└── models/
    ├── __init__.py         # Imports all model modules so @register_model fires
    ├── localvit.py         # LocalityFeedForward, attention, h_swish/SE/ECA, LocalVisionTransformer
    ├── tnt.py              # Vanilla TNT baseline (Attention, Block, PixelEmbed, TNT)
    ├── tnt_moex.py         # TNT + Moment Exchange variant
    ├── localvit_tnt.py     # LNL block (TNT inner/outer attn + LocalityFeedForward)
    ├── localvit_pvt.py     # Locality variant of PVT
    ├── localvit_swin.py    # Locality variant of Swin
    ├── localvit_t2t.py     # Locality variant of T2T-ViT
    ├── deit.py             # DeiT baseline
    ├── pvt.py              # PVT baseline
    ├── swin_transformer.py # Swin baseline
    ├── swin_moex.py        # Swin + MoEx
    ├── t2t_vit.py          # T2T-ViT baseline
    ├── t2t_vit_block.py    # T2T building blocks
    ├── token_transformer.py
    ├── token_performer.py
    ├── modeling_resnet.py  # ResNet hybrid stem
    └── configs.py          # ViT-B/16, ViT-L/16 etc. config dicts (ml_collections)
```

## Architecture

The LNL block (in `LNL.py` / `LNL_MoEx.py`) processes **two streams in parallel** at each depth:

```
LocalViT_TNT
├── PixelEmbed       (Conv2d 7×7 stride=4 → unfold)        # B → B*num_patches, in_dim, h, w
├── Patch projection (Linear: num_pixel*in_dim → embed_dim)
├── N × Block
│   ├── Inner attention   (pixel-level, on in_dim features)
│   ├── Inner MLP         (timm Mlp on in_dim)
│   ├── Project pixel → patch  (norm1_proj + Linear)
│   ├── Outer attention   (patch-level, on embed_dim)
│   └── LocalityFeedForward    # ← KEY DIFFERENCE FROM VANILLA TNT
│       reshape (B, N, C) → (B, C, H, W) → 1×1 conv → 3×3 dwconv → SE/ECA → 1×1 conv → reshape back
└── Linear head
```

Vanilla TNT (`models/tnt.py`) uses `timm.Mlp` for the outer FFN. LNL replaces it with `LocalityFeedForward`, which adds spatial reasoning via depth-wise conv. `LNL_MoEx` swaps in `models/tnt_moex.py` for Moment Exchange feature normalization.

### Model variants

| Variant | embed_dim | in_dim | depth | num_heads | in_num_head |
|---------|-----------|--------|-------|-----------|-------------|
| Ti      | 192       | 12     | 12    | 3         | 3           |
| S       | 384       | 24     | 12    | 6         | 4           |
| B       | 768       | 48     | 12    | 12        | 4           |

Registered functions: `LNL_Ti`, `LNL_S`, `LNL_MoEx_Ti`, `LNL_MoEx_S`, plus `tnt_t/s/b_patch16_224`, `localvit_*` variants, etc.

## Conventions

- **Model registration**: every public model uses `@register_model` (timm). To add a variant, define a function decorated with `@register_model`, set `model.default_cfg`, and ensure the module is imported from `models/__init__.py`.
- **Default config helper**: `_cfg()` at the top of each model file sets ImageNet defaults (`num_classes=1000`, `input_size=(3, 224, 224)`, `crop_pct=0.9`, `interpolation='bicubic'`). LNL/TNT use mean/std `(0.5, 0.5, 0.5)` rather than ImageNet defaults.
- **Stochastic depth**: linearly increasing `DropPath` rate via `torch.linspace(0, drop_path_rate, depth)`.
- **Weight init**: `trunc_normal_(std=.02)` for Linear and position params, constant 0/1 for LayerNorm, applied via `_init_weights`.
- **Shape convention**: sequence tensors are `(B, N, C)`; for conv layers, reshape to `(B, C, H, W)` where `H = W = sqrt(num_token - 1)` (the `-1` is the cls token, which is split off and re-concatenated around the conv block).
- **Authorship comment**: most files start with an `Author: Omid Nejati` docstring — preserve it when editing.

## Training & Evaluation

- Full procedure lives in [Instructions-best.ipynb](file:///d:/Locality-iN-Locality/Instructions-best.ipynb) (optimized training and model checkpoint loading) or [Instructions.ipynb](file:///d:/Locality-iN-Locality/Instructions.ipynb) (original). Key points:

- Standard timm-style training: AdamW + cosine LR + warmup, RandAugment, Mixup (α=0.8), CutMix (α=1.0), label smoothing 0.1.
- ImageNet directory structure: `train/<class>/*.jpg` and `val/<class>/*.jpg`.
- Evaluation reports Top-1 / Top-5 accuracy with center crop (crop_pct=0.9).
- Checkpoints store the full `state_dict` plus optimizer state for resuming.

## Working in This Repo

- **Read before changing**: `LNL.py` and `LNL_MoEx.py` reuse the parent `TNT` class from `models/tnt.py` / `models/tnt_moex.py` — the inner-transformer plumbing (`norm_in`, `attn_in`, `mlp_in`, `proj`) lives in `Block.__init__`. Confirm what the parent provides before duplicating logic.
- **Adding a new model**: copy an existing `Block` + module-level model factory pair, update the table above, register via `@register_model`, and wire the import into `models/__init__.py`.
- **Tensor shape gotchas**: `num_pixel = new_patch_size ** 2` must match the inner-transformer feature length; `Nsqrt = sqrt(N)` for outer reshape assumes the cls token has been split off first.
- **Don't reformat unrelated code**. The codebase mirrors timm style (long lines, terse comments) — match it.

## Available Skills

This repo has Claude Code skills under `.claude/skills/`:

- **`ml-workflow`** — training, evaluation, dataset prep, augmentation, distributed training, experiment tracking
- **`deep-learning`** — architecture implementation, attention/locality mechanisms, debugging, performance optimization, model scaling

Invoke with `/ml-workflow` or `/deep-learning` when those topics come up.
