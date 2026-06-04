"""Vision Transformer + Cross-Attention fusion PPO model.

Architecture:
  1. ViT-style patch embedding + Transformer encoder → visual tokens (per depth image)
  2. Cross-attention: 1D vector features attend to visual tokens
  3. MLP torso → action distribution / value

This is the "next-level" upgrade over the basic CNN encoder.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState

from .debug_logger import dbg


# ---------------------------------------------------------------------------
# Vision Transformer (ViT) for depth images
# ---------------------------------------------------------------------------

class DepthViT(nn.Module):
    """A lightweight ViT that encodes a single depth image into a fixed-size feature."""

    def __init__(
        self,
        in_channels: int = 1,
        img_h: int = 240,
        img_w: int = 640,
        patch_size: int = 16,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert img_h % patch_size == 0 and img_w % patch_size == 0, \
            f"Image dims {img_h}x{img_w} must be divisible by patch_size {patch_size}"
        self.patch_size = patch_size
        self.num_patches = (img_h // patch_size) * (img_w // patch_size)
        self.embed_dim = embed_dim

        # Patch embedding via Conv2d
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        # Row-major position embeddings (learned)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) batch of depth images
        Returns:
            (N, embed_dim) global feature vector (CLS token)
        """
        # Patch embedding: (N, C, H, W) → (N, embed_dim, h', w')
        x = self.patch_embed(x)  # (N, E, h//P, w//P)
        N, E, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # (N, num_patches, E)

        # Add CLS token + position embeddings
        cls_tokens = self.cls_token.expand(N, -1, -1)  # (N, 1, E)
        x = torch.cat([cls_tokens, x], dim=1)          # (N, 1+num_patches, E)
        x = x + self.pos_embed[:, :x.size(1), :]       # broadcast pos embed
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)       # (N, 1+num_patches, E)
        x = self.norm(x[:, 0, :])     # (N, E) — CLS token only
        return x


# ---------------------------------------------------------------------------
# Cross-Attention Fusion: vector features attend to visual tokens
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """Multi-head cross-attention: 1D vector queries attend to 2D visual keys/values."""

    def __init__(
        self,
        query_dim: int,
        visual_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_proj = nn.Linear(query_dim, visual_dim)
        self.norm = nn.LayerNorm(visual_dim)

    def forward(
        self,
        query: torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (N, query_dim)  — 1D vector features
            visual_tokens: (N, num_tokens, visual_dim)  — flattened visual features
        Returns:
            (N, visual_dim) attended features
        """
        q = self.query_proj(query).unsqueeze(1)  # (N, 1, visual_dim)
        attn_out, _ = self.cross_attn(q, visual_tokens, visual_tokens)
        return self.norm(attn_out.squeeze(1))


# ---------------------------------------------------------------------------
# TransformerPPOModel
# ---------------------------------------------------------------------------

class TransformerPPOModel(MLPModel):
    """ViT + Cross-Attention PPO model for depth-based navigation."""

    is_recurrent: bool = False  # can be recurrent in subclasses

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        # ViT params
        vit_patch_size: int = 16,
        vit_embed_dim: int = 128,
        vit_num_heads: int = 4,
        vit_num_layers: int = 3,
        vit_mlp_ratio: float = 4.0,
        vit_dropout: float = 0.1,
        # Cross-attention params
        cross_attn_heads: int = 4,
        cross_attn_dropout: float = 0.1,
        # Shared ViT
        share_vit: bool = True,
    ) -> None:
        # ---- categorize observations ----
        self._obs_groups_2d: list[str] = []
        self._obs_channels_2d: list[int] = []
        obs_img_h: int = 240
        obs_img_w: int = 640

        active: list[str] = obs_groups[obs_set]
        obs_groups_1d: list[str] = []
        for g in active:
            sh = obs[g].shape
            if len(sh) == 4:  # (B, C, H, W)
                self._obs_groups_2d.append(g)
                self._obs_channels_2d.append(cast(int, sh[1]))
                obs_img_h = cast(int, sh[2])
                obs_img_w = cast(int, sh[3])
            elif len(sh) == 2:
                obs_groups_1d.append(g)
            else:
                raise ValueError(f"Unsupported observation shape for '{g}': {sh}")

        self._vit_embed_dim: int = vit_embed_dim

        # ---- delegate to parent (1D observations only) ----
        obs_groups_mod: dict[str, list[str]] = {**obs_groups, obs_set: obs_groups_1d}
        super().__init__(
            obs, obs_groups_mod, obs_set, output_dim,
            hidden_dims, activation, obs_normalization, distribution_cfg,
        )

        # ---- build shared ViT encoder ----
        self._num_2d_groups: int = len(self._obs_groups_2d)
        self.vit: DepthViT | None = None
        self._share_vit: bool = share_vit
        if self._num_2d_groups > 0:
            self.vit = DepthViT(
                in_channels=self._obs_channels_2d[0],
                img_h=obs_img_h,
                img_w=obs_img_w,
                patch_size=vit_patch_size,
                embed_dim=vit_embed_dim,
                num_heads=vit_num_heads,
                num_layers=vit_num_layers,
                mlp_ratio=vit_mlp_ratio,
                dropout=vit_dropout,
            )

        # ---- build cross-attention fusion ----
        self._has_cross_attn: bool = (self._num_2d_groups > 0)
        if self._has_cross_attn:
            self.cross_attn = CrossAttentionFusion(
                query_dim=self.obs_dim,
                visual_dim=vit_embed_dim * max(self._num_2d_groups, 1),
                num_heads=cross_attn_heads,
                dropout=cross_attn_dropout,
            )

        # ---- rebuild MLP to accept fused dim ----
        self._rebuild_mlp(hidden_dims, activation)

    def _rebuild_mlp(
        self,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
    ) -> None:
        """Override MLP with fusion-aware input dim."""
        fused_dim: int = self.obs_dim
        if self._has_cross_attn:
            fused_dim = self._vit_embed_dim * self._num_2d_groups

        act_fn: type[nn.Module] = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        layers: list[nn.Module] = []
        in_dim = fused_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(in_dim, hdim))
            layers.append(act_fn())
            in_dim = hdim
        self.mlp = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """1D normalize → cross-attend with ViT visual features → MLP."""
        latent_1d: torch.Tensor = super().get_latent(obs)

        if self.vit is None or self._num_2d_groups == 0:
            return latent_1d  # no visual inputs

        # Detect padded trajectory shape
        has_time_dim = False
        time_steps: int = 0
        sh = obs[self._obs_groups_2d[0]].shape
        if len(sh) == 5:  # (T, B, C, H, W)
            has_time_dim = True
            time_steps = cast(int, sh[0])

        def flatten_imgs(t: torch.Tensor) -> torch.Tensor:
            return t.flatten(0, 1) if has_time_dim else t

        def unflatten_feat(t: torch.Tensor) -> torch.Tensor:
            return t.unflatten(0, (time_steps, -1)) if has_time_dim else t

        # Encode each depth image through shared ViT
        visual_parts: list[torch.Tensor] = []
        for group_name in self._obs_groups_2d:
            img: torch.Tensor = cast(torch.Tensor, obs[group_name])
            img = flatten_imgs(img)
            feat: torch.Tensor = self.vit(img)      # (N*T, embed_dim)

            # NaN guard
            feat = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
            visual_parts.append(feat)

        # Concatenate all visual features
        visual_feat: torch.Tensor = torch.cat(visual_parts, dim=-1)  # (N*T, num_groups*E)
        visual_feat = unflatten_feat(visual_feat)

        # Cross-attention: 1D query attends to visual
        if has_time_dim:
            # (T, B, D1d) and (T, B, Dvis)
            T, B = latent_1d.shape[0], latent_1d.shape[1]
            latent_1d_flat = latent_1d.reshape(T * B, -1)
            visual_feat_flat = visual_feat.reshape(T * B, -1)
            # Use visual_feat as both K and V (it's already global)
            # MultiheadAttention expects (N, S, E) for K/V, we expand to (N, 1, E)
            visual_tokens = visual_feat_flat.unsqueeze(1)  # (T*B, 1, E)
            # Cross-attn is applied via the fusion module
            fused_flat = self.cross_attn(latent_1d_flat, visual_tokens)
            fused = fused_flat.reshape(T, B, -1)
        else:
            visual_tokens = visual_feat.unsqueeze(1)  # (B, 1, E)
            fused = self.cross_attn(latent_1d, visual_tokens)  # (B, E)

        return fused

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward with NaN guard."""
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)
        mlp_output = torch.nan_to_num(mlp_output, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.distribution is not None:
            self.distribution.update(mlp_output)
            if stochastic_output:
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)

        # Critic path: log periodically
        if not hasattr(self, '_fwd_cnt'):
            self._fwd_cnt = 0
        self._fwd_cnt += 1
        if self._fwd_cnt % 500 == 1:
            v = mlp_output.detach()
            if v.numel() > 0:
                dbg(f"[TransPPO] critic #{self._fwd_cnt}: n={v.numel()} mean={v.mean().item():.3f} std={v.std(unbiased=False).item():.3f} nan={torch.isnan(v).any().item()}")
        return mlp_output

    def _get_latent_dim(self) -> int:
        return self._vit_embed_dim * max(self._num_2d_groups, 1)
