"""CNN PPO model for depth-image obstacle avoidance.

Processes front + back depth images through a shared CNN encoder,
concatenates the flattened features with standard 1D observations,
and feeds everything into an MLP torso.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState


# ---------------------------------------------------------------------------
# Helper: ModuleDict subclass so PyLance knows __getitem__ returns ModuleDict
# ---------------------------------------------------------------------------

class CNNModuleDict(nn.ModuleDict):
    """ModuleDict whose __getitem__ is typed to return ModuleDict."""

    def __getitem__(self, key: str) -> nn.ModuleDict:
        return cast(nn.ModuleDict, super().__getitem__(key))


# ---------------------------------------------------------------------------
# CNNPPOModel
# ---------------------------------------------------------------------------

class CNNPPOModel(MLPModel):
    """Depth-image CNN + MLP model for obstacle-aware navigation."""

    cnns: CNNModuleDict

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
        cnn_channels: list[int] | None = None,
        cnn_kernel: int = 3,
        cnn_stride: int = 2,
        cnn_padding: int = 1,
        share_cnn: bool = True,
    ) -> None:
        # ---- categorize observations (1D vs 2D) ----
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

        # ---- pre-compute CNN output dim ----
        if cnn_channels is None:
            cnn_channels = [8, 16]
        self._cnn_feat_dim: int = 0
        seen_cnn = False
        for _i in range(len(self.obs_groups_2d)):
            if share_cnn and seen_cnn:
                continue
            seen_cnn = True
            cout: int = cnn_channels[-1] if cnn_channels else 1
            self._cnn_feat_dim += cout

        # ---- delegate to parent ----
        obs_groups_mod: dict[str, list[str]] = {**obs_groups, obs_set: obs_groups_1d}
        super().__init__(
            obs, obs_groups_mod, obs_set, output_dim,
            hidden_dims, activation, obs_normalization, distribution_cfg,
        )

        # ---- build CNN modules ----
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
            cnn_mod = CNNModuleDict({"conv": nn.Sequential(*seq)})
            if share_cnn:
                shared_cnn = cnn_mod
            cnns_dict[group_name] = cnn_mod

        self.cnns = CNNModuleDict(cnns_dict)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """1D normalize → cat with CNN encodings → MLP."""
        latent_1d: torch.Tensor = super().get_latent(obs)

        # Detect padded trajectory shape
        has_time_dim = False
        time_steps: int = 0
        if self.obs_groups_2d:
            sh = obs[self.obs_groups_2d[0]].shape
            if len(sh) == 5:  # (T, B, C, H, W)
                has_time_dim = True
                time_steps = cast(int, sh[0])

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

            feat: torch.Tensor = mod["conv"](img)

            if has_time_dim:
                feat = feat.unflatten(0, (time_steps, -1))
            cnn_parts.append(feat)

        if cnn_parts:
            latent_cnn: torch.Tensor = torch.cat(cnn_parts, dim=-1)
            return torch.cat([latent_1d, latent_cnn], dim=-1)
        return latent_1d

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self._cnn_feat_dim
