# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configuration for NaviSpotRL task."""

from isaaclab.managers import ActionTermCfg


def get_action_cfg() -> ActionTermCfg:
    """Return action configuration.
    
    Action space: [left_wheel_velocity, right_wheel_velocity]
    """
    return ActionTermCfg(
        terms={
            "robot_action": {
                "type": "manager_based",
                "action_dim": 2,
                "params": {},
            },
        }
    )