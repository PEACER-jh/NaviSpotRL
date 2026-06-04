"""LSTM / GRU PPO models for waypoint-following local planning.

These models extend ``rsl_rl.models.MLPModel`` with a recurrent head that processes
a sequence of waypoint vectors before the MLP torso.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState

from .debug_logger import dbg


class _RecurrentPPOModel(MLPModel):
    """Recurrent PPO base: RNN → MLP → output.

    Handles both:
      - Single-step ``(B, obs_dim)`` — ``act`` phase (uses ``self._hidden``)
      - Padded trajectory ``(T, B, obs_dim)`` — ``update`` phase (uses
        ``hidden_state`` arg from PPO)

    Set ``rnn_type="lstm"`` or ``rnn_type="gru"`` in the config.
    """

    is_recurrent: bool = True
    rnn: nn.Module  # LSTM or GRU

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
    ) -> None:
        self._rnn_hidden_dim: int = rnn_hidden_dim
        super().__init__(
            obs, obs_groups, obs_set, output_dim,
            hidden_dims, activation, obs_normalization, distribution_cfg,
        )
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                self.obs_dim, rnn_hidden_dim, rnn_num_layers, batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                self.obs_dim, rnn_hidden_dim, rnn_num_layers, batch_first=True,
            )
        self._hidden: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Normalize observations → RNN → MLP input."""
        latent: torch.Tensor = super().get_latent(obs)

        has_time_dim = (latent.dim() == 3)  # (T, B, D) → update phase
        if has_time_dim:
            latent = latent.transpose(0, 1)  # → (B, T, D)
        else:
            latent = latent.unsqueeze(1)      # → (B, 1, D)

        # Safety: clip extreme values before RNN
        latent = torch.nan_to_num(latent, nan=0.0, posinf=1.0, neginf=-1.0)

        # Use provided hidden_state (update) or self._hidden (act)
        h = hidden_state if hidden_state is not None else self._hidden
        rnn_out: torch.Tensor
        rnn_out, new_h = self.rnn(latent, h)
        rnn_out = torch.nan_to_num(rnn_out, nan=0.0, posinf=1.0, neginf=-1.0)

        if has_time_dim:
            return rnn_out.transpose(0, 1)  # (B, T, H) → (T, B, H)
        self._hidden = new_h
        return rnn_out.squeeze(1)  # (B, 1, H) → (B, H)

    # ------------------------------------------------------------------
    #  RNN state management
    # ------------------------------------------------------------------

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Zero out RNN hidden state for finished episodes."""
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
        """Detach RNN hidden state for truncated BPTT."""
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
        return self._rnn_hidden_dim



    def forward(
        self,
        obs: torch.Tensor | TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward with NaN guard. Flattens T dim for distribution."""
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)
        # Periodic NaN check
        if not hasattr(self, '_fwd_cnt'):
            self._fwd_cnt = 0
        self._fwd_cnt += 1
        do_log = (self._fwd_cnt % 500 == 1)
        if do_log:
            for name, p in self.named_parameters():
                if torch.isnan(p).any():
                    dbg(f"[RNN] ERROR NaN in param: {name}")
        if do_log and torch.isnan(mlp_output).any():
            dbg("[RNN] ERROR NaN in mlp_output before distribution")
        mlp_output = torch.nan_to_num(mlp_output, nan=0.0, posinf=1.0, neginf=-1.0)
        if self.distribution is not None:
            has_time = (mlp_output.dim() == 3)
            T_v, B_v = mlp_output.shape[:2] if has_time else (0, mlp_output.shape[0])
            if has_time:
                mlp_output = mlp_output.reshape(T_v * B_v, -1)
            self.distribution.update(mlp_output)
            if stochastic_output:
                out = self.distribution.sample()
                if has_time:
                    out = out.reshape(T_v, B_v, -1)
                return out
            out = self.distribution.deterministic_output(mlp_output)
            if has_time:
                out = out.reshape(T_v, B_v, -1)
            return out
        # Critic: periodic summary only
        if do_log:
            v = mlp_output.detach()
            if v.numel() > 0:
                dbg(f"[RNN] critic #{self._fwd_cnt}: n={v.numel()} mean={v.mean().item():.3f} std={v.std(unbiased=False).item():.3f} nan={torch.isnan(v).any().item()}")
            else:
                dbg(f"[RNN] critic #{self._fwd_cnt}: EMPTY")
        return mlp_output
        return mlp_output
LSTMPPOModel = _RecurrentPPOModel
GRUPPOModel = _RecurrentPPOModel
