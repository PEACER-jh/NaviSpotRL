# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resume training from a saved checkpoint — no Hydra, just argparse + manual config."""

import argparse
import sys
import os
import time
import logging
from datetime import datetime

from isaaclab.app import AppLauncher

# ---- argparse (all our own, no cli_args / Hydra) ----
parser = argparse.ArgumentParser(description="Resume RL training from a checkpoint.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--resume_path", type=str, required=True, help="Path to checkpoint .pt file")
parser.add_argument("--seed", type=int, default=42)

# AppLauncher / render args (provides --device, --headless, --enable_cameras, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# force enable_cameras to false unless explicitly set in headless mode
if not getattr(args_cli, 'enable_cameras', False):
    args_cli.enable_cameras = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest."""

import gymnasium as gym
import torch
os.environ["NAVISPOTRL_MODE"] = "train"

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

logger = logging.getLogger(__name__)
import NaviSpotRL.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # ---- Load configs directly (no Hydra) ----
    env_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device
    agent_cfg.seed = args_cli.seed

    # Handle deprecated config fields (distribution_cfg -> stochastic etc.)
    import importlib.metadata as metadata
    rsl_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_version)

    # ---- Checkpoint path ----
    checkpoint_path = os.path.abspath(args_cli.resume_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"[INFO] Resuming from checkpoint: {checkpoint_path}")

    # ---- Log directory (same run folder) ----
    log_dir = os.path.dirname(checkpoint_path)
    print(f"[INFO] Logging to: {log_dir}")
    env_cfg.log_dir = log_dir

    # ---- Create env ----
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---- Create runner ----
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # ---- Load checkpoint ----
    print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
    runner.load(checkpoint_path)

    # Compute remaining iterations (rsl_rl learn(N) means "train N MORE iterations")
    if args_cli.max_iterations is not None:
        start_it = runner.current_learning_iteration  # e.g. 2100
        remaining = max(1, args_cli.max_iterations - start_it)
        agent_cfg.max_iterations = remaining
        total = start_it + remaining
        print(f"[INFO] Resuming from iter {start_it}, will train {remaining} more -> total {total}")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    start_time = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
