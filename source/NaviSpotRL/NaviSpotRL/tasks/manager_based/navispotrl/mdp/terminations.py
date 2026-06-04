# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination configuration for NaviSpotRL task."""

from isaaclab.managers import TerminationTermCfg


def get_termination_cfg() -> TerminationTermCfg:
    """Return termination configuration.
    
    Termination conditions:
    - time_out: 超时结束
    - reached_goal: 到达所有路径点
    - collision: 碰撞障碍物
    """
    return TerminationTermCfg(
        terms={
            "time_out": {
                "type": "manager_based",
                "params": {},
            },
            "reached_goal": {
                "type": "manager_based",
                "params": {},
            },
            "collision": {
                "type": "manager_based",
                "params": {},
            },
        }
    )