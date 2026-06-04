# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a custom four-wheel differential drive rover with panoramic cameras."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

PANORAMIC_ROVER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(os.path.dirname(__file__), "urdf", "rover.usd"),
    ),
    actuators={
        "left_wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "left_wheel_back_joint"],
            velocity_limit=50.0,     # 限速 10 rad/s
            stiffness=0.0,           # 不设弹簧刚度
            damping=0.5,             # 添加阻尼，让速度变化更平滑
        ),
        "right_wheels": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel_joint", "right_wheel_back_joint"],
            velocity_limit=50.0,
            stiffness=0.0,
            damping=0.5,
        ),
    },
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.04),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)