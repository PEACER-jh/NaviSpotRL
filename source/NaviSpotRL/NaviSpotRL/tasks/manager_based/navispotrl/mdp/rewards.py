# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward configuration for NaviSpotRL task."""

from isaaclab.managers import RewardTermCfg


def get_reward_cfg() -> RewardTermCfg:
    """Return reward configuration.
    
    Reward terms:
    - progress: 奖励靠近目标点 (weight: 5.0)
    - reached: 奖励到达目标点 (weight: 10.0)  
    - collision: 惩罚碰撞障碍物 (weight: -20.0)
    - proximity: 惩罚靠近障碍物 (weight: -1.0)
    - angular_velocity: 惩罚急转弯，促进平滑轨迹 (weight: -0.1)
    """
    return RewardTermCfg(
        terms={
            "progress": {
                "type": "manager_based",
                "weight": 5.0,
                "params": {},
            },
            "reached": {
                "type": "manager_based",
                "weight": 10.0,
                "params": {},
            },
            "collision": {
                "type": "manager_based",
                "weight": -20.0,
                "params": {},
            },
            "proximity": {
                "type": "manager_based",
                "weight": -1.0,
                "params": {},
            },
            "angular_velocity": {
                "type": "manager_based",
                "weight": -0.1,
                "params": {},
            },
        }
    )