import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from typing import Optional, Sequence, Tuple, Type, Union
from torch.nn import LayerNorm
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
import torch.utils.checkpoint as checkpoint
__all__ = [
    "SwinTransformer3D",
    "window_partition_3d",
    "window_reverse_3d",
    "WindowAttention3D",
    "SwinTransformerBlock3D",
    "PatchMerging3D",
    "MERGING_MODE_3D",
    "BasicLayer3D",
]


def window_partition_3d(x, window_size):
    """3D window partition operation"""
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0],
        window_size[0],
        H // window_size[1],
        window_size[1],
        W // window_size[2],
        window_size[2],
        C
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, window_size[0] * window_size[1] * window_size[2], C)
    return windows


def window_reverse_3d(windows, window_size, dims):
    """Reverse 3D window operation"""
    B, D, H, W = dims
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, -1)
    return x


def get_window_size_3d(x_size, window_size, shift_size=None):
    """Computing window size for 3D input"""
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """3D Window based multi-head self attention with relative position bias"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: Sequence[int],
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(B_ // nw, nw, self.num_heads, N, N) + mask.to(attn.dtype).unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """3D Swin Transformer block"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: Sequence[int],
            shift_size: Sequence[int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            act_layer: str = "GELU",
            norm_layer: Type[LayerNorm] = nn.LayerNorm,
            use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer,
                       dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size_3d((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)

        # Padding for spatial dimensions only
        pad_d0 = pad_h0 = pad_w0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_h1 = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_w1 = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_w0, pad_w1, pad_h0, pad_h1, pad_d0, pad_d1))

        _, Dp, Hp, Wp, _ = x.shape
        dims = [B, Dp, Hp, Wp]

        # Cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # Window partition
        x_windows = window_partition_3d(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse_3d(attn_windows, window_size, dims)

        # Reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        # Remove padding
        if pad_d1 > 0 or pad_h1 > 0 or pad_w1 > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)

        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging3D(nn.Module):
    """3D Patch merging layer"""

    def __init__(
            self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3, c_multiplier: int = 2
    ) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, c_multiplier * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        B, D, H, W, C = x.shape
        x = torch.cat(
            [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))],
            dim=-1
        )
        x = self.norm(x)
        x = self.reduction(x)
        return x


MERGING_MODE_3D = {"mergingv2": PatchMerging3D}


def compute_mask_3d(dims, window_size, shift_size, device):
    """Computing region masks for 3D input"""
    D, H, W = dims
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0

    for d in [slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)]:
        for h in [slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)]:
            for w in [slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)]:
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition_3d(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class BasicLayer3D(nn.Module):
    """Basic 3D Swin Transformer layer for one stage"""

    def __init__(
            self,
            dim: int,
            depth: int,
            num_heads: int,
            window_size: Sequence[int],
            drop_path: list,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            norm_layer: Type[LayerNorm] = nn.LayerNorm,
            c_multiplier: int = 2,
            downsample: Optional[nn.Module] = None,
            use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)
        ])

        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, spatial_dims=3, c_multiplier=c_multiplier
            )

    def forward(self, x):
        B, C, D, H, W = x.size()
        window_size, shift_size = get_window_size_3d((D, H, W), self.window_size, self.shift_size)
        x = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]

        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]

        attn_mask = compute_mask_3d([Dp, Hp, Wp], window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)

        x = x.permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        return x


class PositionalEmbedding3D(nn.Module):
    """3D Absolute positional embedding"""

    def __init__(self, dim: int, patch_dim: tuple) -> None:
        super().__init__()
        self.dim = dim
        self.patch_dim = patch_dim
        D, H, W = patch_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, D, H, W))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed


class PatchEmbed3D(nn.Module):
    """3D image to patch embedding"""

    def __init__(
            self,
            img_size: Tuple[int, int, int],
            patch_size: Sequence[int],
            in_chans: int,
            embed_dim: int,
            norm_layer: Optional[Type[nn.Module]] = None,
            flatten: bool = False,
            spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)
        # 保持通道在第二维，空间维度在后续
        return x


class BasicLayer3D_FullAttention(nn.Module):
    """3D Full Attention Layer"""

    def __init__(
            self,
            dim: int,
            depth: int,
            num_heads: int,
            window_size: Sequence[int],
            drop_path: list,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            norm_layer: Type[LayerNorm] = nn.LayerNorm,
            c_multiplier: int = 2,
            downsample: Optional[nn.Module] = None,
            use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=self.no_shift,  # 无移位
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)
        ])

        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, spatial_dims=3, c_multiplier=c_multiplier
            )

    def forward(self, x):
        B, C, D, H, W = x.size()
        # 修改后（正确）
        window_size = get_window_size_3d((D, H, W), self.window_size, None)
        x = x.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]

        attn_mask = None  # 全注意力无需掩码

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)

        x = x.permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]
        return x


class SwinTransformer3D(nn.Module):
    """3D Swin Transformer for DTI data"""

    def __init__(
            self,
            img_size: Tuple[int, int, int] = (96, 96, 96),
            in_chans: int = 1,
            embed_dim: int = 48,
            window_size: Sequence[int] = (7, 7, 7),
            first_window_size: Sequence[int] = (7, 7, 7),
            patch_size: Sequence[int] = (4, 4, 4),
            depths: Sequence[int] = (2, 2, 6, 2),
            num_heads: Sequence[int] = (3, 6, 12, 24),
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            norm_layer: Type[LayerNorm] = nn.LayerNorm,
            patch_norm: bool = True,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
            c_multiplier: int = 2,
            last_layer_full_MSA: bool = False,
            downsample="mergingv2",
            num_classes=2,
            to_float: bool = False,
            **kwargs,
    ) -> None:
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.first_window_size = first_window_size
        self.patch_size = patch_size
        self.to_float = to_float

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            spatial_dims=spatial_dims
        )

        grid_size = self.patch_embed.grid_size
        self.grid_size = grid_size
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Positional embeddings
        self.pos_embeds = nn.ModuleList()
        patch_dim = grid_size
        pos_embed_dim = embed_dim

        for i in range(self.num_layers):
            self.pos_embeds.append(PositionalEmbedding3D(pos_embed_dim, patch_dim))
            pos_embed_dim = pos_embed_dim * c_multiplier
            patch_dim = (max(1, patch_dim[0] // 2), max(1, patch_dim[1] // 2), max(1, patch_dim[2] // 2))

        # Build layers
        self.layers = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE_3D) if isinstance(downsample, str) else downsample

        # First layer with potentially different window size
        layer = BasicLayer3D(
            dim=int(embed_dim),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=self.first_window_size,
            drop_path=dpr[sum(depths[:0]): sum(depths[:1])],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            c_multiplier=c_multiplier,
            downsample=down_sample_mod if 0 < self.num_layers - 1 else None,
            use_checkpoint=use_checkpoint,
        )
        self.layers.append(layer)

        # Middle layers
        for i_layer in range(1, self.num_layers - 1):
            layer = BasicLayer3D(
                dim=int(embed_dim * (c_multiplier ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[:i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=down_sample_mod if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # Last layer with optional full attention
        if not last_layer_full_MSA:
            layer = BasicLayer3D(
                dim=int(embed_dim * c_multiplier ** (self.num_layers - 1)),
                depth=depths[self.num_layers - 1],
                num_heads=num_heads[self.num_layers - 1],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:self.num_layers - 1]): sum(depths[:self.num_layers])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
        else:
            last_window_size = (
                self.grid_size[0] // int(2 ** (self.num_layers - 1)),
                self.grid_size[1] // int(2 ** (self.num_layers - 1)),
                self.grid_size[2] // int(2 ** (self.num_layers - 1)),
            )

            layer = BasicLayer3D_FullAttention(
                dim=int(embed_dim * c_multiplier ** (self.num_layers - 1)),
                depth=depths[self.num_layers - 1],
                num_heads=num_heads[self.num_layers - 1],
                window_size=last_window_size,
                drop_path=dpr[sum(depths[:self.num_layers - 1]): sum(depths[:self.num_layers])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.num_features = int(embed_dim * c_multiplier ** (self.num_layers - 1))

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.output_dim = self.num_features
    def forward(self, x):
        if self.to_float:
            x = x.float()

        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for i in range(self.num_layers):
            x = self.pos_embeds[i](x)
            x = self.layers[i](x.contiguous())

        # Classification head
        # x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        # x = self.norm(x)
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        # x = self.head(x)

        return x    #4 192 2 2 2
