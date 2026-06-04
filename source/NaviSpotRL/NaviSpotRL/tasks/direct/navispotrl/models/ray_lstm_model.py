"""RayEncoder + LSTM + MLP — combined model for obstacle-aware navigation.

ARCHITECTURE:
  ┌─ lidar_rays (64D) ──┐
  │                       ├──→ RayEncoder → spatial_feat (128D) ──┐
  └─ pose_cond (9D) ────┘                                         │
  ┌─ policy+waypoints (9D) ──→ 2-layer LSTM → temporal_feat(128D) ┤
  │                                                                 │
  └──────────────────── concat(256D) ──→ MLP → action(2D) ────────┘

KEY INSIGHT:
  - LSTM (130K params) is the PRIMARY decision-maker for navigation timing.
  - RayEncoder (80K params) is the SPATIAL advisor for obstacle awareness.
  - Pose queries the rays: "given my state, which direction should I worry about?"
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils.utils import unpad_trajectories

from .debug_logger import dbg
from .ray_encoder import RayEncoder


class RayLSTMModel(MLPModel):
    """RayEncoder + LSTM + MLP for obstacle-aware navigation."""

    is_recurrent: bool = True
    ray_encoder: RayEncoder
    rnn: nn.LSTM
    drive_encoder: nn.Sequential
    fusion: nn.Sequential

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        # ---- RayEncoder-specific ----
        ray_hidden: int = 32,
        ray_d_model: int = 64,
        ray_num_queries: int = 4,
        ray_num_heads: int = 4,
        ray_output_dim: int = 128,
        # ---- LSTM-specific ----
        rnn_hidden_dim: int = 128,
        rnn_num_layers: int = 2,
    ) -> None:
        # ---- Identify obs groups ----
        self.ray_group = "lidar_rays"
        active: list[str] = obs_groups[obs_set]
        # 1D groups (policy + waypoints) — everything except lidar_rays
        obs_1d = [g for g in active if g != self.ray_group]
        obs_groups_mod: dict[str, list[str]] = {**obs_groups, obs_set: obs_1d}

        self._rnn_hidden_dim = rnn_hidden_dim
        self._ray_output_dim = ray_output_dim

        # ---- Build parent MLP (with 1D obs only) ----
        super().__init__(
            obs, obs_groups_mod, obs_set, output_dim,
            hidden_dims, activation, obs_normalization, distribution_cfg,
        )

        # ---- RayEncoder: lidar_rays + pose_cond → spatial features ----
        pose_dim = self.obs_dim  # policy(5) + waypoints(4) = 9
        self.ray_encoder = RayEncoder(
            ray_dim=obs[self.ray_group].shape[-1],
            pose_dim=pose_dim,
            hidden=ray_hidden,
            d_model=ray_d_model,
            num_queries=ray_num_queries,
            num_heads=ray_num_heads,
            output_dim=ray_output_dim,
        )

        # ---- Driving state encoder: 1D obs → RNN input ----
        self.drive_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ELU(),
            nn.Linear(64, rnn_hidden_dim),
            nn.ELU(),
        )

        # ---- LSTM: temporal understanding of navigation ----
        self.rnn = nn.LSTM(
            rnn_hidden_dim, rnn_hidden_dim, rnn_num_layers, batch_first=True,
        )
        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None

        # ---- Fusion: adjust MLP input dim to accept ray features ----
        fusion_in = rnn_hidden_dim + ray_output_dim  # 128 + 128 = 256
        first_linear: nn.Linear = self.mlp[0]  # type: ignore[assignment]
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, first_linear.in_features),
            nn.ELU(),
        )

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """
        Process:
          1. Extract 1D obs via parent → encode → LSTM → temporal_feat
          2. RayEncoder(lidar_rays, 1D obs as pose_cond) → spatial_feat
          3. Concat → fusion → MLP input
        """
        # ---- 1D driving observation (for LSTM + pose query) ----
        latent_1d: torch.Tensor = super().get_latent(obs)  # (B, obs_dim) or (T, B, obs_dim)

        has_time_dim = latent_1d.dim() == 3
        time_steps: int = latent_1d.shape[0] if has_time_dim else 1

        if has_time_dim:
            latent_1d_flat = latent_1d.flatten(0, 1)  # (T*B, obs_dim)
        else:
            latent_1d_flat = latent_1d

        # Encode for LSTM
        drive_feat: torch.Tensor = self.drive_encoder(latent_1d_flat)  # (T*B, rnn_hidden)
        if has_time_dim:
            drive_feat = drive_feat.unflatten(0, (time_steps, -1))  # (T, B, rnn_hidden)

        # ---- LSTM: process temporal sequence ----
        if has_time_dim:
            rnn_input = drive_feat.transpose(0, 1)  # (B, T, H)
        else:
            rnn_input = drive_feat.unsqueeze(1)  # (B, 1, H)

        h = hidden_state if hidden_state is not None else self._hidden
        rnn_out, new_h = self.rnn(rnn_input, h)
        rnn_out = torch.nan_to_num(rnn_out, nan=0.0, posinf=1.0, neginf=-1.0)

        if has_time_dim:
            rnn_out = rnn_out.transpose(0, 1)  # (T, B, H)
        else:
            rnn_out = rnn_out.squeeze(1)  # (B, H)

        # ---- RayEncoder: spatial understanding ----
        rays: torch.Tensor = obs[self.ray_group]  # (B, 64) or (T, B, 64)
        if has_time_dim:
            rays_flat = rays.flatten(0, 1)  # (T*B, 64)
            pose_for_ray = latent_1d_flat     # (T*B, 9)
        else:
            rays_flat = rays
            pose_for_ray = latent_1d_flat

        spatial_feat = self.ray_encoder(rays_flat, pose_for_ray)  # (T*B, ray_output_dim)
        if has_time_dim:
            spatial_feat = spatial_feat.unflatten(0, (time_steps, -1))  # (T, B, 128)

        # ---- Fusion ----
        if has_time_dim:
            rnn_for_fusion = rnn_out.flatten(0, 1)
            spatial_for_fusion = spatial_feat.flatten(0, 1)
        else:
            rnn_for_fusion = rnn_out
            spatial_for_fusion = spatial_feat

        fused = torch.cat([rnn_for_fusion, spatial_for_fusion], dim=-1)
        fused = self.fusion(fused)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=10.0, neginf=-10.0)

        if has_time_dim and masks is not None:
            fused = fused.unflatten(0, (time_steps, -1))
            fused = unpad_trajectories(fused, masks)
        elif has_time_dim:
            fused = fused.reshape(-1, fused.shape[-1])

        if not has_time_dim:
            self._hidden = new_h
        return fused

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)
        mlp_output = torch.nan_to_num(mlp_output, nan=0.0, posinf=1.0, neginf=-1.0)
        if self.distribution is not None:
            self.distribution.update(mlp_output)
            if stochastic_output:
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    # ------------------------------------------------------------------
    #  LSTM state management
    # ------------------------------------------------------------------

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if self._hidden is None:
            return
        if isinstance(self._hidden, tuple):
            h, c = self._hidden
            if dones is not None:
                mask = (1.0 - dones.float()).view(1, -1, 1)
                self._hidden = (h * mask, c * mask)
        elif dones is not None:
            mask = (1.0 - dones.float()).view(1, -1, 1)
            self._hidden = self._hidden * mask

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self._hidden is None:
            return
        self.reset(dones)
        if isinstance(self._hidden, tuple):
            h_d, c_d = self._hidden
            self._hidden = (h_d.detach(), c_d.detach())
        else:
            self._hidden = self._hidden.detach()

    def get_hidden_state(self) -> HiddenState:
        return self._hidden

    def _get_latent_dim(self) -> int:
        # Must not depend on self.mlp (not yet created during __init__).
        # Fusion layer maps (rnn_hidden + ray_output) → mlp[0].in_features.
        # mlp[0].in_features = hidden_dims[0] passed to parent.
        return self._rnn_hidden_dim + getattr(self, '_ray_output_dim', 128)
