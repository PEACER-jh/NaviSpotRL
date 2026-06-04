# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING, field
from typing import Any

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlMLPModelCfg,
    RslRlPpoAlgorithmCfg,
)


# ==================== Distribution ====================

_GAUSSIAN = RslRlMLPModelCfg.GaussianDistributionCfg(
    class_name="GaussianDistribution",
    init_std=0.25,  # v64: moderate noise — prevent collapse while avoiding explosion
)

# ==================== Model definitions ====================

_MODEL_PREFIX = "NaviSpotRL.tasks.direct.navispotrl.models"


def _make_cfg(base_cls: type, **kwargs: object) -> Any:
    """Create a base_cls instance and attach extra kwargs as plain attributes."""
    known = {
        "class_name", "hidden_dims", "activation", "obs_normalization",
        "distribution_cfg", "stochastic", "init_noise_std",
        "noise_std_type", "state_dependent_std",
    }
    base_kwargs = {k: v for k, v in kwargs.items() if k in known}
    cfg = base_cls(**base_kwargs)
    for k, v in kwargs.items():
        if k not in known:
            setattr(cfg, k, v)
    return cfg


# --- MLP baseline ---
MLP_ACTOR = _make_cfg(
    RslRlMLPModelCfg,
    class_name="MLPModel",
    hidden_dims=[128, 64],         # 更大的容量处理导航
    activation="elu",
    distribution_cfg=_GAUSSIAN,
)
MLP_CRITIC = _make_cfg(
    RslRlMLPModelCfg,
    class_name="MLPModel",
    hidden_dims=[128, 64],
    activation="elu",
)

# --- RayCNN (1D CNN for lidar_rays + MLP torso) ---
RAY_CNN_ACTOR = _make_cfg(
    RslRlMLPModelCfg,
    class_name=f"{_MODEL_PREFIX}.ray_cnn_model.RayCNNModel",
    hidden_dims=[128, 64],
    activation="elu",
    ray_channels=[8, 16, 32],
    ray_kernel=5,
    ray_feat_dim=64,
    distribution_cfg=_GAUSSIAN,
)
RAY_CNN_CRITIC = _make_cfg(
    RslRlMLPModelCfg,
    class_name=f"{_MODEL_PREFIX}.ray_cnn_model.RayCNNModel",
    hidden_dims=[128, 64],
    activation="elu",
    ray_channels=[8, 16, 32],
    ray_kernel=5,
    ray_feat_dim=64,
)

# --- CNN-only (no RNN) ---
CNN_ACTOR = _make_cfg(
    RslRlMLPModelCfg,
    class_name=f"{_MODEL_PREFIX}.cnn_ppo_model.CNNPPOModel",
    hidden_dims=[64, 32],
    activation="elu",
    cnn_channels=[8, 16],
    cnn_kernel=3,
    cnn_stride=2,
    cnn_padding=1,
    share_cnn=True,
    distribution_cfg=_GAUSSIAN,
)
CNN_CRITIC = _make_cfg(
    RslRlMLPModelCfg,
    class_name=f"{_MODEL_PREFIX}.cnn_ppo_model.CNNPPOModel",
    hidden_dims=[64, 32],
    activation="elu",
    cnn_channels=[8, 16],
    cnn_kernel=3,
    cnn_stride=2,
    cnn_padding=1,
    share_cnn=True,
)

# --- CNN + LSTM ---
CNN_LSTM_ACTOR = _make_cfg(
    RslRlMLPModelCfg,
    class_name=f"{_MODEL_PREFIX}.cnn_rnn_ppo_model.CNNRNNPPOModel",
    # hidden_dims set per-model above
    activation="elu",
    rnn_type="lstm",
    rnn_hidden_dim=64,
    rnn_num_layers=1,
    cnn_channels=[8, 16],
    cnn_kernel=3,
    cnn_stride=2,
    cnn_padding=1,
    share_cnn=True,
    distribution_cfg=_GAUSSIAN,
)
CNN_LSTM_CRITIC = _make_cfg(
    RslRlMLPModelCfg,
    class_name=f"{_MODEL_PREFIX}.cnn_rnn_ppo_model.CNNRNNPPOModel",
    # hidden_dims set per-model above
    activation="elu",
    rnn_type="lstm",
    rnn_hidden_dim=64,
    rnn_num_layers=1,
    cnn_channels=[8, 16],
    cnn_kernel=3,
    cnn_stride=2,
    cnn_padding=1,
    share_cnn=True,
)

# --- CNN + GRU ---
CNN_GRU_ACTOR = _make_cfg(
    RslRlMLPModelCfg,
    class_name=f"{_MODEL_PREFIX}.cnn_rnn_ppo_model.CNNRNNPPOModel",
    hidden_dims=[64, 32],
    activation="elu",
    rnn_type="gru",
    rnn_hidden_dim=64,
    rnn_num_layers=1,
    cnn_channels=[8, 16],
    cnn_kernel=3,
    cnn_stride=2,
    cnn_padding=1,
    share_cnn=True,
    distribution_cfg=_GAUSSIAN,
)
CNN_GRU_CRITIC = _make_cfg(
    RslRlMLPModelCfg,
    class_name=f"{_MODEL_PREFIX}.cnn_rnn_ppo_model.CNNRNNPPOModel",
    hidden_dims=[64, 32],
    activation="elu",
    rnn_type="gru",
    rnn_hidden_dim=64,
    rnn_num_layers=1,
    cnn_channels=[8, 16],
    cnn_kernel=3,
    cnn_stride=2,
    cnn_padding=1,
    share_cnn=True,
)

# --- RNN-only (no CNN) ---
RNN_ACTOR = _make_cfg(
    RslRlMLPModelCfg,
    class_name=f"{_MODEL_PREFIX}.lstm_ppo_model._RecurrentPPOModel",
    hidden_dims=[64, 32],
    activation="elu",
    rnn_type="lstm",
    rnn_hidden_dim=64,
    rnn_num_layers=1,
    distribution_cfg=_GAUSSIAN,
)
RNN_CRITIC = _make_cfg(
    RslRlMLPModelCfg,
    class_name=f"{_MODEL_PREFIX}.lstm_ppo_model._RecurrentPPOModel",
    hidden_dims=[64, 32],
    activation="elu",
    rnn_type="lstm",
    rnn_hidden_dim=64,
    rnn_num_layers=1,
)

# --- Transformer (ViT + Cross-Attention) ---
TRANS_ACTOR = _make_cfg(
    RslRlMLPModelCfg,
    class_name=f"{_MODEL_PREFIX}.transformer_ppo_model.TransformerPPOModel",
    # hidden_dims set per-model above
    activation="elu",
    vit_patch_size=16,
    vit_embed_dim=128,
    vit_num_heads=4,
    vit_num_layers=3,
    vit_mlp_ratio=4.0,
    vit_dropout=0.1,
    cross_attn_heads=4,
    cross_attn_dropout=0.1,
    share_vit=True,
    distribution_cfg=_GAUSSIAN,
)
TRANS_CRITIC = _make_cfg(
    RslRlMLPModelCfg,
    class_name=f"{_MODEL_PREFIX}.transformer_ppo_model.TransformerPPOModel",
    # hidden_dims set per-model above
    activation="elu",
    vit_patch_size=16,
    vit_embed_dim=128,
    vit_num_heads=4,
    vit_num_layers=3,
    vit_mlp_ratio=4.0,
    vit_dropout=0.1,
    cross_attn_heads=4,
    cross_attn_dropout=0.1,
    share_vit=True,
)

# --- RayEncoder + LSTM (v62) ---
RAY_LSTM_ACTOR = _make_cfg(
    RslRlMLPModelCfg,
    class_name=f"{_MODEL_PREFIX}.ray_lstm_model.RayLSTMModel",
    hidden_dims=[128, 64],
    activation="elu",
    obs_normalization=True,           # normalize obs to zero-mean unit-variance
    ray_hidden=32,
    ray_d_model=64,
    ray_num_queries=4,
    ray_num_heads=4,
    ray_output_dim=128,
    rnn_hidden_dim=128,
    rnn_num_layers=2,
    distribution_cfg=_GAUSSIAN,
)
RAY_LSTM_CRITIC = _make_cfg(
    RslRlMLPModelCfg,
    class_name=f"{_MODEL_PREFIX}.ray_lstm_model.RayLSTMModel",
    hidden_dims=[128, 64],
    activation="elu",
    obs_normalization=True,
    ray_hidden=32,
    ray_d_model=64,
    ray_num_queries=4,
    ray_num_heads=4,
    ray_output_dim=128,
    rnn_hidden_dim=128,
    rnn_num_layers=2,
)

# ==================== Runner ====================

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32  # 32 robots × 32 = 1024 steps per iteration
    max_iterations = 20000
    save_interval = 50
    experiment_name = "cartpole_direct"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    obs_groups: dict[str, list[str]] = {
        "actor": ["policy", "waypoints", "lidar_rays"],
        "critic": ["policy", "waypoints", "lidar_rays"],
    }

    # ---- active model (MLP baseline) ----
    actor = RAY_LSTM_ACTOR;  critic = RAY_LSTM_CRITIC  # RayEncoder + LSTM (v62)

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.012,            # v67: ↓ was 0.025 — stronger signal, less forced exploration
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-4,          # adaptive schedule handles scaling
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,               # v67: ↑ was 0.01 — allow larger updates with 3× reward
        max_grad_norm=2.0,             # v67: ↑ was 1.0 — prevent clipping with larger gradients
    )
