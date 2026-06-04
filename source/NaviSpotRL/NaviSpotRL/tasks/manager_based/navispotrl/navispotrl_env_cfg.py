# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for NaviSpotRL task."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from NaviSpotRL.tasks.manager_based.navispotrl.mdp.scene import NavispotrlSceneCfg
from NaviSpotRL.tasks.manager_based.navispotrl.mdp.simulation import get_simulation_cfg
from NaviSpotRL.tasks.manager_based.navispotrl.mdp.observations import get_observation_cfg
from NaviSpotRL.tasks.manager_based.navispotrl.mdp.actions import get_action_cfg
from NaviSpotRL.tasks.manager_based.navispotrl.mdp.rewards import get_reward_cfg
from NaviSpotRL.tasks.manager_based.navispotrl.mdp.terminations import get_termination_cfg


@configclass
class NavispotrlEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the NaviSpotRL environment."""

    # Scene
    scene: NavispotrlSceneCfg = NavispotrlSceneCfg(
        num_envs=32,
        env_spacing=4.0,
        replicate_physics=True
    )

    # Simulation
    sim = get_simulation_cfg()

    # Episode
    decimation: int = 2
    episode_length_s: float = 15.0

    # Observations
    observations = get_observation_cfg()

    # Actions
    actions = get_action_cfg()

    # Rewards
    rewards = get_reward_cfg()

    # Terminations
    terminations = get_termination_cfg()

    # Navigation parameters
    target_reach_threshold: float = 0.3
    num_waypoints: int = 5

    # Obstacles
    num_obstacles: int = 4
    obstacle_radius: float = 0.2
    obstacle_height: float = 1.0

    # Robot
    dof_names = ["left_wheel_joint", "right_wheel_joint"]