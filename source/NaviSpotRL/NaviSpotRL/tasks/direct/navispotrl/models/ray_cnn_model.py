"""RayCNN PPO model — lightweight 1D CNN for lidar_rays + MLP torso.

Separates the obstacle-ray processing from the policy head:
  lidar_rays (B, 2, N/2) → 1D CNN → ray_features (B, 32)
                                  ↓ concat
  policy + waypoints (B, 9) ─────→ 41D → MLP → action

This is the minimal CNN variant — 2 Conv1d layers followed by an MLP.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState


class RayCNNModel(MLPModel):
    """MLP with a tiny 1D CNN front-end for lidar_rays."""

    ray_cnn: nn.ModuleDict

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
        # ---- RayCNN-specific ----
        ray_channels: list[int] | None = None,
        ray_kernel: int = 5,
        ray_feat_dim: int = 32,
    ) -> None:
        self.ray_group = "lidar_rays"
        self.ray_feat_dim = int(ray_feat_dim)

        # ---- Split obs groups: lidar_rays handled by CNN, rest by MLP ----
        active: list[str] = obs_groups[obs_set]
        obs_1d = [g for g in active if g != self.ray_group]
        obs_groups_mod: dict[str, list[str]] = {**obs_groups, obs_set: obs_1d}

        # ---- Build parent MLP with reduced input ----
        super().__init__(
            obs, obs_groups_mod, obs_set, output_dim,
            hidden_dims, activation, obs_normalization, distribution_cfg,
        )

        # ---- Build tiny 1D CNN for lidar_rays ----
        ray_dim = obs[self.ray_group].shape[-1]  # 64
        # Reshape to (B, 2, 32) for Conv1d over rays
        self.ray_shape = (2, ray_dim // 2)  # (2, 32)

        if ray_channels is None:
            ray_channels = [4, 8]  # tiny: 2→4→8 channels

        in_ch = self.ray_shape[0]  # 2
        seq: list[nn.Module] = []
        for ch in ray_channels:
            seq.extend([
                nn.Conv1d(in_ch, ch, ray_kernel, padding=ray_kernel // 2),
                nn.ELU(),
            ])
            in_ch = ch
        seq.append(nn.AdaptiveAvgPool1d(1))
        seq.append(nn.Flatten())
        self.ray_cnn = nn.ModuleDict({"conv": nn.Sequential(*seq)})

        # Verify output size
        dummy = torch.randn(1, self.ray_shape[0], self.ray_shape[1])
        cnn_out = self.ray_cnn["conv"](dummy).shape[-1]
        assert cnn_out == ray_channels[-1], \
            f"CNN output dim {cnn_out} != last channel {ray_channels[-1]}"

        # Extra linear to project CNN features to ray_feat_dim
        self.ray_proj = nn.Linear(ray_channels[-1], self.ray_feat_dim)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Process lidar_rays with CNN, then concat with MLP latent."""
        # Standard 1D latent from policy + waypoints
        latent_1d: torch.Tensor = super().get_latent(obs)

        # Process lidar_rays with CNN
        rays: torch.Tensor = cast(torch.Tensor, obs[self.ray_group])
        # Handle padded recurrent shape (T, B, ...)
        if rays.dim() == 3:  # (T, B, D)
            T = rays.shape[0]
            rays = rays.flatten(0, 1)
        else:
            T = 0

        # Reshape to (B, 2, 32) for Conv1d
        rays = rays.view(-1, self.ray_shape[0], self.ray_shape[1])

        # CNN forward
        ray_feat = self.ray_cnn["conv"](rays)  # (B, channels)
        ray_feat = self.ray_proj(ray_feat)     # (B, ray_feat_dim)

        if T > 0:
            ray_feat = ray_feat.unflatten(0, (T, -1))

        return torch.cat([latent_1d, ray_feat], dim=-1)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.ray_feat_dim
