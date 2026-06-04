# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene configuration for NaviSpotRL task."""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg

from NaviSpotRL.robots.jetbot import JETBOT_CONFIG


@configclass
class NavispotrlSceneCfg(InteractiveSceneCfg):
    """Configuration for the NaviSpotRL scene."""
    
    robot: ArticulationCfg = JETBOT_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    
    num_envs: int = 32
    env_spacing: float = 4.0
    replicate_physics: bool = True