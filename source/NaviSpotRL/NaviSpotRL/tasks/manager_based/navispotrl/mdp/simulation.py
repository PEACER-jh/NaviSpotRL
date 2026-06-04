# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation configuration for NaviSpotRL task."""

from isaaclab.sim import SimulationCfg


def get_simulation_cfg() -> SimulationCfg:
    """Return simulation configuration."""
    return SimulationCfg(
        dt=1/120,
        render_interval=2,
    )