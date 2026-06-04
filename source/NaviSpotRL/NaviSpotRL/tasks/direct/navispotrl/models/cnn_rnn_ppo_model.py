"""CNN + RNN PPO model — v2: Navigation-prioritized architecture.

ARCHITECTURE:
  ┌─ policy obs (dot,cross,dist) + waypoints ──┐
  │  (7D vector → small MLP → 32D)              │
  │                                              ├──→ cat(64D) → MLP head → action
  ├─ LSTM(drive_state, 32D hidden) ────────────┘
  │  (processes temporal sequence of driving)
  │
  └─ CNN(depth_front, depth_back) → 32D feature ─→ gate signal (mild fusion)
     (visual: used ONLY as gating, not primary decision)

KEY INSIGHT: The 7D driving observation (dot, cross, dist, waypoints) is the
PRIMARY signal for navigation. Depth images are an AUXILIARY signal — they
provide obstacle context but should NOT dominate the model's capacity.

The RNN processes the DRIVING state (not the full visual) over time, so it
learns "turn → face target → drive" from a 7D temporal sequence.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils.utils import unpad_trajectories

from .debug_logger import dbg

from .cnn_ppo_model import CNNModuleDict


class CNNRNNPPOModel(MLPModel):
    """v2: CNN processes depth → small feature. RNN processes DRIVING obs only."""

    is_recurrent: bool = True
    cnns: CNNModuleDict
    rnn: nn.Module

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
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 128,
        rnn_num_layers: int = 1,
        cnn_channels: list[int] | None = None,
        cnn_kernel: int = 3,
        cnn_stride: int = 2,
        cnn_padding: int = 1,
        share_cnn: bool = True,
    ) -> None:
        # ---- Categorize observations ----
        self.obs_groups_2d: list[str] = []
        self.obs_dims_2d: list[tuple[int, int]] = []
        self.obs_channels_2d: list[int] = []

        active: list[str] = obs_groups[obs_set]
        obs_groups_1d: list[str] = []
        for g in active:
            sh = obs[g].shape
            if len(sh) == 4:
                self.obs_groups_2d.append(g)
                self.obs_dims_2d.append((cast(int, sh[2]), cast(int, sh[3])))
                self.obs_channels_2d.append(cast(int, sh[1]))
            elif len(sh) == 2:
                obs_groups_1d.append(g)
            else:
                raise ValueError(f"Unsupported observation shape for '{g}': {sh}")

        # ---- Pre-compute CNN output dim (small, squeezed hard) ----
        if cnn_channels is None:
            cnn_channels = [8, 16]
        self._cnn_out_dim: int = 32  # COMPACT: CNN → 32D vector only

        self._rnn_hidden_dim: int = rnn_hidden_dim

        # ---- Pre-compute MLP input dim for _get_latent_dim ----
        self._fusion_out_dim = hidden_dims[0] if isinstance(hidden_dims, (list, tuple)) else cast(int, hidden_dims)

        # ---- Delegate 1D obs to parent ----
        obs_groups_mod: dict[str, list[str]] = {**obs_groups, obs_set: obs_groups_1d}
        super().__init__(
            obs, obs_groups_mod, obs_set, output_dim,
            hidden_dims, activation, obs_normalization, distribution_cfg,
        )

        # ---- Build compact CNN encoder ----
        cnns_dict: dict[str, CNNModuleDict] = {}
        shared_cnn: CNNModuleDict | None = None
        for i, group_name in enumerate(self.obs_groups_2d):
            if share_cnn and shared_cnn is not None:
                cnns_dict[group_name] = shared_cnn
                continue
            cin: int = self.obs_channels_2d[i]
            seq: list[nn.Module] = []
            for cout in cnn_channels:
                seq.extend([
                    nn.Conv2d(cin, cout, cnn_kernel, cnn_stride, cnn_padding),
                    nn.ELU(),
                ])
                cin = cout
            seq.append(nn.AdaptiveAvgPool2d((1, 1)))
            seq.append(nn.Flatten())
            # Final linear projection: CNN features → 32D compact vector
            raw_cnn_dim = cnn_channels[-1] if cnn_channels else 1
            seq.append(nn.Linear(raw_cnn_dim, self._cnn_out_dim))
            seq.append(nn.ELU())
            cnn_mod = CNNModuleDict({"conv": nn.Sequential(*seq)})
            if share_cnn:
                shared_cnn = cnn_mod
            cnns_dict[group_name] = cnn_mod

        self.cnns = CNNModuleDict(cnns_dict)

        # ---- Driving state encoder: 1D obs → RNN input ----
        # Small MLP to embed the sparse 1D state before RNN
        drive_dim = self.obs_dim  # typically 7: [dot, cross, dist, wp1_x, wp1_y, wp2_x, wp2_y]
        self.drive_encoder = nn.Sequential(
            nn.Linear(drive_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
        )

        # ---- RNN: processes driving state over time ----
        rnn_input_dim: int = 64  # compact driving encoding
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                rnn_input_dim, rnn_hidden_dim, rnn_num_layers, batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                rnn_input_dim, rnn_hidden_dim, rnn_num_layers, batch_first=True,
            )

        # ---- Fusion layer: RNN hidden + CNN feature → MLP input ----
        self.fusion = nn.Sequential(
            nn.Linear(rnn_hidden_dim + self._cnn_out_dim, self._fusion_out_dim),
            nn.ELU(),
        )

        self._hidden: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None

    # ------------------------------------------------------------------
    #  Forward / latent
    # ------------------------------------------------------------------

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """1D driving state → encoder → RNN → fuse with CNN → output."""
        # Empty batch guard
        first_key = next(iter(obs.keys()))
        first_val: torch.Tensor = cast(torch.Tensor, obs[first_key])
        if first_val.numel() == 0:
            return first_val.new_zeros((0, cast(int, self.mlp[0].in_features)))

        # ---- 1D driving observation (the PRIMARY signal) ----
        # Use parent's raw 1D extraction (before parent MLP)
        latent_1d: torch.Tensor = super().get_latent(obs)  # (B, obs_dim) or (T, B, obs_dim)

        # Detect time dimension
        has_time_dim = latent_1d.dim() == 3
        time_steps: int = latent_1d.shape[0] if has_time_dim else 1

        if has_time_dim:
            latent_1d = latent_1d.flatten(0, 1)  # (T*B, obs_dim)
        # Encode driving state
        drive_feat: torch.Tensor = self.drive_encoder(latent_1d)  # (T*B, 32)
        if has_time_dim:
            drive_feat = drive_feat.unflatten(0, (time_steps, -1))  # (T, B, 32)

        # ---- CNN: encode depth images → compact feature ----
        cnn_parts: list[torch.Tensor] = []
        seen_mod_ids: set[int] = set()
        for group_name in self.obs_groups_2d:
            mod = cast(CNNModuleDict, self.cnns[group_name])
            if id(mod) in seen_mod_ids:
                continue
            seen_mod_ids.add(id(mod))

            img: torch.Tensor = cast(torch.Tensor, obs[group_name])
            if has_time_dim:
                img = img.flatten(0, 1)

            feat = mod["conv"](img)
            feat = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)

            if has_time_dim:
                feat = feat.unflatten(0, (time_steps, -1))
            cnn_parts.append(feat)

        cnn_feat: torch.Tensor
        if cnn_parts:
            cnn_feat = torch.cat(cnn_parts, dim=-1)  # (T*B, 32) or (T, B, 32)
        else:
            cnn_feat = torch.zeros(*drive_feat.shape[:-1], self._cnn_out_dim, device=drive_feat.device)

        # ---- RNN: process driving sequence ----
        # RNN expects (B, T, D)
        if has_time_dim:
            rnn_input = drive_feat.transpose(0, 1)  # (B, T, 32)
        else:
            rnn_input = drive_feat.unsqueeze(1)  # (B, 1, 32)

        h = hidden_state if hidden_state is not None else self._hidden
        rnn_out: torch.Tensor
        rnn_out, new_h = self.rnn(rnn_input, h)
        rnn_out = torch.nan_to_num(rnn_out, nan=0.0, posinf=1.0, neginf=-1.0)

        if has_time_dim:
            rnn_out = rnn_out.transpose(0, 1)  # (T, B, H)
            cnn_for_fusion = cnn_feat  # (T, B, 32)
        else:
            rnn_out = rnn_out.squeeze(1)  # (B, H)
            cnn_for_fusion = cnn_feat.squeeze(1) if cnn_feat.dim() == 3 else cnn_feat

        # ---- Fusion: RNN hidden + CNN feature → MLP input ----
        fused = torch.cat([rnn_out, cnn_for_fusion], dim=-1)  # (T, B, H+32) or (B, H+32)
        fused = self.fusion(fused)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=10.0, neginf=-10.0)

        if has_time_dim and masks is not None:
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
        """Forward with NaN guard."""
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)
        mlp_output = torch.nan_to_num(mlp_output, nan=0.0, posinf=1.0, neginf=-1.0)
        if self.distribution is not None:
            self.distribution.update(mlp_output)
            if stochastic_output:
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def _get_latent_dim(self) -> int:
        return self._fusion_out_dim

    # ------------------------------------------------------------------
    #  RNN state management
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

    def as_jit(self) -> nn.Module:
        return self

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return self
