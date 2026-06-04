# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation configuration for NaviSpotRL task."""

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg


def get_observation_cfg() -> ObservationGroupCfg:
    """Return observation configuration.
    
    Observation space: [dot_to_target, cross_to_target, distance_to_target]
    """
    return ObservationGroupCfg(
        concatenate_terms=True,
        enable_corruption=False,
        history_length=1,
        flatten_history_dim=True,
        terms={
            "robot_obs": ObservationTermCfg(
                func=None,
                params={},
            ),
        },
    )