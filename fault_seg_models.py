#!/usr/bin/env python3
"""
fault_seg_models.py
===================

Two encoder-decoder networks of matched capacity (about 25 M parameters) for
seismic fault segmentation:

    UNet        convolutional baseline [1]
    SwinUNETR   Swin Transformer encoder [3] with a convolutional decoder [2];
                the architecture behind the best reported results on Thebe [4]

Each network maps a standardized single-channel section (B, 1, H, W) to a
fault-probability map of the same size. Inputs are reflect-padded to a multiple
of 32 and the output is cropped back, so any section size is accepted.

Matched between the two networks
--------------------------------
* six-level resolution pyramid (strides 1, 2, 4, 8, 16, 32), five skip connections
* channel pattern (w, w, 2w, 4w, 8w, 16w)
* decoder topology: 2x2 transposed convolution -> concatenation with the skip ->
  convolution block, and a 1x1 convolution followed by a sigmoid
* BatchNorm and LeakyReLU(0.01) in every convolutional block
* parameter count: UNet (base_width=58) 25.7 M, SwinUNETR (feature_size=48) 25.2 M

Different between the two networks
----------------------------------
* encoder: stacked 3x3 convolutions with max pooling (U-Net) vs. shifted-window
  self-attention with patch merging (Swin Transformer)
* convolution blocks: double convolution (U-Net) vs. residual double convolution
  (Swin UNETR), which also passes its encoder features through residual adapters
  (all except the stride-16 feature)

Swin UNETR configuration: patch size 2, feature size 48, depths 2-2-2-2, heads
3-6-12-24, window size 8, MLP ratio 4. The structure follows the MONAI reference
implementation (Apache-2.0) of [2], with these adaptations: window size 8 instead
of 7, BatchNorm instead of InstanceNorm in the convolution blocks, and
truncated-normal initialization of the Linear layers.

Input size
----------
Trained on 256 x 256 crops. At inference a section is processed in a single
forward pass with internal padding; multiples of 128 need no window padding in
the deepest Swin stage. Requires PyTorch >= 2.1.

Usage
-----
    from fault_seg_models import build_model
    model = build_model("unet")          # or "swin_unetr"
    prob = model(x)                      # x: (B, 1, H, W) -> fault probability, same shape

    python fault_seg_models.py                       # self-test + parameter/MAC table
    python fault_seg_models.py --benchmark           # AdamW step time + peak GPU memory (bf16)

References
----------
[1] Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks
    for Biomedical Image Segmentation. MICCAI 2015, LNCS 9351, 234-241.
[2] Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H. R., Xu, D. (2022). Swin
    UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI
    Images. arXiv:2201.01266.
[3] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.
    (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted
    Windows. ICCV 2021, 10012-10022.
[4] Zhang, Z., Chen, R., Ma, J. (2024). Improving Seismic Fault Recognition with
    Self-Supervised Pre-Training: A Study of 3D Transformer-Based with
    Multi-Scale Decoding and Fusion. Remote Sensing 16(5), 922.
    doi:10.3390/rs16050922
"""

from __future__ import annotations

import argparse
import time
from functools import lru_cache
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNet", "SwinUNETR", "build_model", "count_params", "count_macs"]

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

DIVISOR = 32  # total stride of both networks (2 ** 5)


def _act() -> nn.Module:
    return nn.LeakyReLU(0.01, inplace=True)


def _pad_to_multiple(x: torch.Tensor, m: int = DIVISOR):
    """Reflect-pad H and W (bottom/right) to a multiple of m. Returns (x, (H, W))."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph == 0 and pw == 0:
        return x, (h, w)
    return F.pad(x, (0, pw, 0, ph), mode="reflect"), (h, w)


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by normalization and LeakyReLU (U-Net block [1])."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), _act(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), _act(),
        )

    def forward(self, x):
        return self.block(x)


class ResConvBlock(nn.Module):
    """Residual double-convolution block of Swin UNETR [2]; the shortcut is a 1x1
    convolution when the number of channels changes."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.act = _act()
        self.norm1 = nn.BatchNorm2d(cout)
        self.norm2 = nn.BatchNorm2d(cout)
        self.shortcut = cin != cout
        if self.shortcut:
            self.conv3 = nn.Conv2d(cin, cout, 1, bias=False)
            self.norm3 = nn.BatchNorm2d(cout)

    def forward(self, x):
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        res = self.norm3(self.conv3(x)) if self.shortcut else x
        return self.act(out + res)


class UpBlock(nn.Module):
    """Decoder step shared by both networks: 2x2 transposed convolution (stride 2) ->
    concatenation with the skip -> conv block."""

    def __init__(self, cin: int, skip: int, cout: int, block: type[nn.Module]):
        super().__init__()
        self.transp_conv = nn.ConvTranspose2d(cin, cout, 2, stride=2, bias=False)
        self.conv_block = block(cout + skip, cout)

    def forward(self, x, skip):
        x = self.transp_conv(x)
        return self.conv_block(torch.cat([x, skip], dim=1))


# --------------------------------------------------------------------------- #
# 1) U-Net baseline
# --------------------------------------------------------------------------- #


class UNet(nn.Module):
    """
    U-Net [1] with a six-level pyramid (five 2x2 max poolings).

    Widths (w, w, 2w, 4w, 8w, 16w) at strides (1, 2, 4, 8, 16, 32). base_width=58
    gives 25.7 M parameters, against 25.2 M for the Swin UNETR (feature_size=48). The
    theoretical receptive field at the bottleneck is 284 pixels.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_width: int = 58):
        super().__init__()
        w = base_width
        widths = (w, w, 2 * w, 4 * w, 8 * w, 16 * w)
        self.widths = widths
        chans = (in_channels,) + widths
        self.encoder = nn.ModuleList(
            [DoubleConv(chans[i], chans[i + 1]) for i in range(6)]
        )
        self.pool = nn.MaxPool2d(2)
        self.decoder = nn.ModuleList(
            [UpBlock(widths[i], widths[i - 1], widths[i - 1], DoubleConv)
             for i in range(5, 0, -1)]
        )
        self.out = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x):
        x, (h, w) = _pad_to_multiple(x)
        skips = []
        for i, enc in enumerate(self.encoder):
            x = enc(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()                      # bottleneck, stride 32
        for dec in self.decoder:
            x = dec(x, skips.pop())
        return torch.sigmoid(self.out(x)[..., :h, :w].float())   # fault probability (fp32)


# --------------------------------------------------------------------------- #
# 2) Swin UNETR
# --------------------------------------------------------------------------- #


class DropPath(nn.Module):
    """Stochastic depth (per sample)."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """(B, H, W, C) -> (B * nW, ws * ws, C)"""
    b, h, w, c = x.shape
    x = x.reshape(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)


def window_reverse(win: torch.Tensor, ws: int, b: int, h: int, w: int) -> torch.Tensor:
    """(B * nW, ws * ws, C) -> (B, H, W, C)"""
    c = win.shape[-1]
    x = win.reshape(b, h // ws, w // ws, ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)


@lru_cache(maxsize=64)
def _shift_mask(hp: int, wp: int, ws: int, sh: int, sw: int, device: torch.device):
    """Additive shifted-window mask (nW, N, N): 0 for token pairs that were neighbours
    before the cyclic shift, -100 otherwise [3]."""

    def region(n: int, s: int) -> torch.Tensor:
        if s == 0:
            return torch.zeros(n, dtype=torch.long)
        idx = torch.arange(n)
        return (idx >= n - ws).long() + (idx >= n - s).long()

    ids = region(hp, sh)[:, None] * 3 + region(wp, sw)[None, :]
    ids = ids.to(device).reshape(1, hp, wp, 1).float()
    mw = window_partition(ids, ws).squeeze(-1)             # (nW, N)
    diff = mw[:, None, :] - mw[:, :, None]                 # (nW, N, N)
    return torch.zeros_like(diff).masked_fill(diff != 0, -100.0)


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with learned relative position bias [3]."""

    def __init__(self, dim: int, heads: int, window: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window - 1) ** 2, heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)  # (2, N)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window - 1
        rel[:, :, 1] += window - 1
        rel[:, :, 0] *= 2 * window - 1
        self.register_buffer("relative_position_index", rel.sum(-1), persistent=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None):
        bn, n, c = x.shape
        qkv = self.qkv(x).reshape(bn, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)                       # (bn, heads, N, N)
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        attn = attn + bias.reshape(n, n, -1).permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.reshape(bn // nw, nw, self.heads, n, n) + mask[None, :, None]
            attn = attn.reshape(bn, self.heads, n, n)
        attn = self.attn_drop(attn.softmax(dim=-1)).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(bn, n, c)
        return self.proj_drop(self.proj(x))


class SwinBlock(nn.Module):
    """Swin Transformer block [3]: LayerNorm, (shifted-)window attention and a two-layer
    MLP, each with a residual connection. Input (B, H, W, C)."""

    def __init__(self, dim: int, heads: int, window: int, shifted: bool,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.window, self.shifted = window, shifted
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, window, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = x.shape
        ws = self.window
        y = self.norm1(x)
        pad_b, pad_r = (-h) % ws, (-w) % ws            # pad up to a multiple of the window
        if pad_b or pad_r:
            y = F.pad(y, (0, 0, 0, pad_r, 0, pad_b))
        hp, wp = h + pad_b, w + pad_r
        # as in the reference: an axis that already fits in one window is not shifted
        sh = ws // 2 if (self.shifted and h > ws) else 0
        sw = ws // 2 if (self.shifted and w > ws) else 0
        mask = None
        if sh or sw:
            y = torch.roll(y, shifts=(-sh, -sw), dims=(1, 2))
            mask = _shift_mask(hp, wp, ws, sh, sw, y.device)
        y = self.attn(window_partition(y, ws), mask)
        y = window_reverse(y, ws, b, hp, wp)
        if sh or sw:
            y = torch.roll(y, shifts=(sh, sw), dims=(1, 2))
        if pad_b or pad_r:
            y = y[:, :h, :w, :].contiguous()
        return y

    def forward(self, x):
        x = x + self.drop_path(self._attention(x))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    """2x2 neighbours -> concat (4C) -> LayerNorm -> Linear(4C -> 2C) [3]."""

    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):                                # (B, H, W, C)
        b, h, w, c = x.shape
        if h % 2 or w % 2:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                       x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        return self.reduction(self.norm(x))


class SwinStage(nn.Module):
    """`depth` Swin blocks (alternating regular / shifted windows) + patch merging."""

    def __init__(self, dim: int, depth: int, heads: int, window: int, drop_path: Sequence[float],
                 mlp_ratio: float, qkv_bias: bool, drop: float, attn_drop: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, heads, window, shifted=(i % 2 == 1), mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i])
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim)

    def forward(self, x):                                # (B, C, H, W) -> (B, 2C, H/2, W/2)
        x = x.permute(0, 2, 3, 1)
        for blk in self.blocks:
            x = blk(x)
        return self.downsample(x).permute(0, 3, 1, 2)


def _channel_ln(x: torch.Tensor) -> torch.Tensor:
    """Parameter-free channel-wise LayerNorm, applied to each Swin output before the adapters."""
    return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2)


class SwinEncoder(nn.Module):
    """
    Swin Transformer encoder [3]: a stride-2 convolution embeds the input as tokens,
    followed by four stages of (shifted-)window attention blocks, each ending in patch
    merging. A window of `window` tokens per side spans window x s pixels at stride s
    (16 to 128 px for window 8). Returns 5 feature maps at strides 2, 4, 8, 16, 32.
    """

    def __init__(self, in_channels: int, embed_dim: int, depths: Sequence[int],
                 num_heads: Sequence[int], window: int, patch_size: int = 2,
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, drop: float = 0.0,
                 attn_drop: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.pos_drop = nn.Dropout(drop)
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.layers = nn.ModuleList()
        for i, depth in enumerate(depths):
            self.layers.append(SwinStage(
                dim=embed_dim * 2 ** i, depth=depth, heads=num_heads[i], window=window,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop))

    def forward(self, x):
        x = self.pos_drop(self.patch_embed(x))
        outs = [_channel_ln(x)]
        for layer in self.layers:
            x = layer(x)
            outs.append(_channel_ln(x))
        return outs


class SwinUNETR(nn.Module):
    """
    Swin UNETR [2]: Swin Transformer encoder [3] with residual-convolution adapters and
    decoder, six-level pyramid (strides 1 ... 32). Encoder features reach the decoder
    through adapters at strides 1, 2, 4, 8 and 32; the stride-16 Swin feature is passed
    on directly.

    Defaults: feature size 48, patch size 2, depths 2-2-2-2, heads 3-6-12-24, window 8,
    MLP ratio 4.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, feature_size: int = 48,
                 depths: Sequence[int] = (2, 2, 2, 2), num_heads: Sequence[int] = (3, 6, 12, 24),
                 window_size: int = 8, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0):
        super().__init__()
        f = feature_size
        self.swinViT = SwinEncoder(in_channels, f, depths, num_heads, window_size,
                                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                                   attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate)
        R = ResConvBlock
        self.encoder1 = R(in_channels, f)            # stride 1  (raw-input skip)
        self.encoder2 = R(f, f)                      # stride 2
        self.encoder3 = R(2 * f, 2 * f)              # stride 4
        self.encoder4 = R(4 * f, 4 * f)              # stride 8
        self.encoder10 = R(16 * f, 16 * f)           # stride 32 (bottleneck)
        self.decoder5 = UpBlock(16 * f, 8 * f, 8 * f, R)   # -> stride 16, skip = Swin stride-16 feature
        self.decoder4 = UpBlock(8 * f, 4 * f, 4 * f, R)    # -> stride 8
        self.decoder3 = UpBlock(4 * f, 2 * f, 2 * f, R)    # -> stride 4
        self.decoder2 = UpBlock(2 * f, f, f, R)            # -> stride 2
        self.decoder1 = UpBlock(f, f, f, R)                # -> stride 1
        self.out = nn.Conv2d(f, out_channels, 1)
        self.apply(self._init_linear)

    @staticmethod
    def _init_linear(m: nn.Module):
        if isinstance(m, nn.Linear):                 # Swin-style init [3]
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x, (h, w) = _pad_to_multiple(x)
        hs = self.swinViT(x)                         # strides 2, 4, 8, 16, 32
        enc0 = self.encoder1(x)
        enc1 = self.encoder2(hs[0])
        enc2 = self.encoder3(hs[1])
        enc3 = self.encoder4(hs[2])
        dec4 = self.encoder10(hs[4])
        dec3 = self.decoder5(dec4, hs[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        return torch.sigmoid(self.out(out)[..., :h, :w].float())   # fault probability (fp32)


# --------------------------------------------------------------------------- #
# Factory + utilities
# --------------------------------------------------------------------------- #

MODELS = {"unet": UNet, "swin_unetr": SwinUNETR}


def build_model(name: str, **kwargs) -> nn.Module:
    """build_model('unet') or build_model('swin_unetr'); kwargs go to the constructor."""
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'; choose from {list(MODELS)}")
    return MODELS[name](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def count_macs(model: nn.Module, size=(1, 1, 256, 256)) -> float:
    """Multiply-accumulates of one forward pass (conv + matmul ops; FLOPs / 2)."""
    from torch.utils.flop_counter import FlopCounterMode
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    with FlopCounterMode(display=False) as fc:
        model(torch.zeros(size, device=dev))
    model.train(was_training)
    return fc.get_total_flops() / 2


PEAK_LR = {"unet": 1e-3, "swin_unetr": 3e-4}   # peak step sizes of the training runs


def _loss(p: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary cross-entropy + Dice loss at equal weights, computed in fp32."""
    dice = 1 - (2 * (y * p).sum() + eps) / (y.sum() + p.sum() + eps)
    return F.binary_cross_entropy(p, y) + dice


def _benchmark(name: str, crop: int, batch: int, steps: int = 20):
    """Time AdamW training steps in bf16 autocast and report peak GPU memory."""
    dev = torch.device("cuda")
    model = build_model(name).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR[name], betas=(0.9, 0.999),
                            weight_decay=0.01)
    x = torch.randn(batch, 1, crop, crop, device=dev)
    y = (torch.rand(batch, 1, crop, crop, device=dev) > 0.98).float()
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 5):
        if i == 5:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x)
        loss = _loss(p, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    print(f"  {name:11s} batch {batch} @ {crop}x{crop} (bf16): "
          f"{dt * 1000:6.1f} ms/step, {batch / dt:6.1f} img/s, "
          f"peak memory {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB")
    del model, opt
    torch.cuda.empty_cache()


def _self_test(crop: int):
    torch.manual_seed(0)
    print(f"\n{'model':12s} {'params (M)':>11s} {'GMACs @ ' + str(crop):>14s}   output-shape checks")
    counts = {}
    for name in MODELS:
        m = build_model(name)
        n = count_params(m) / 1e6
        counts[name] = n
        g = count_macs(m, (1, 1, crop, crop)) / 1e9
        ok = True
        m.train()
        for shape in [(2, 1, crop, crop), (1, 1, 300, 450), (1, 1, 128, 128)]:
            x = torch.randn(*shape)
            y = m(x)
            ok &= tuple(y.shape) == (shape[0], 1, shape[2], shape[3])
        # every parameter must receive a gradient
        m.zero_grad()
        m(torch.randn(2, 1, crop, crop)).mean().backward()
        missing = [k for k, p in m.named_parameters() if p.grad is None]
        print(f"{name:12s} {n:11.2f} {g:14.1f}   shapes {'OK' if ok else 'FAIL'}, "
              f"grads {'OK' if not missing else 'MISSING ' + str(missing[:3])}")
    print(f"\nparameter ratio unet / swin_unetr = {counts['unet'] / counts['swin_unetr']:.3f}")


def main():
    ap = argparse.ArgumentParser(description="U-Net vs Swin UNETR: self-test and GPU benchmark")
    ap.add_argument("--crop", type=int, default=256, help="training crop size (default 256)")
    ap.add_argument("--batch", type=int, default=16, help="batch size for --benchmark")
    ap.add_argument("--benchmark", action="store_true", help="time a training step and report peak GPU memory")
    args = ap.parse_args()
    _self_test(args.crop)
    if args.benchmark:
        if not torch.cuda.is_available():
            print("\n--benchmark needs a CUDA GPU; none found.")
            return
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        for name in MODELS:
            _benchmark(name, args.crop, args.batch)


if __name__ == "__main__":
    main()
