# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_apply

from .navispotrl_env_cfg import NavispotrlEnvCfg


class NaviSpotRLEnv(ManagerBasedRLEnv):
    """Environment for NaviSpotRL task: navigate through waypoints to reach a final goal,
    avoiding obstacles along the way."""

    cfg: NavispotrlEnvCfg

    def __init__(self, cfg: NavispotrlEnvCfg, render_mode: Optional[str] = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ========== 路径点导航变量 ==========
        self._waypoints = torch.zeros(
            (self.num_envs, self.cfg.num_waypoints, 2),
            device=self.device
        )
        self._current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._num_waypoints = self.cfg.num_waypoints

        # 当前目标点
        self._target_pos = torch.zeros((self.num_envs, 2), device=self.device)

        # 障碍物位置列表
        self._obstacle_positions = torch.zeros(
            (self.num_envs, self.cfg.num_obstacles, 2),
            device=self.device
        )

        # 用于距离追踪（奖励计算用）
        self._prev_distance = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        """Set up the scene: ground, robot, obstacles."""
        # 地面
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # 机器人
        self._robot = Articulation(self.cfg.scene.robot)

        # 障碍物（圆柱体）
        self._obstacles = []
        for i in range(self.cfg.num_obstacles):
            obstacle_cfg = RigidObjectCfg(
                prim_path=f"/World/Obstacle_{i:02d}",
                spawn=sim_utils.CylinderCfg(
                    radius=self.cfg.obstacle_radius,
                    height=self.cfg.obstacle_height,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            )
            obstacle = RigidObject(cfg=obstacle_cfg)
            self._obstacles.append(obstacle)

        # 克隆环境
        self.scene.clone_environments(copy_from_source=False)

        # 添加到场景
        self.scene.articulations["robot"] = self._robot
        for i, obs in enumerate(self._obstacles):
            self.scene.rigid_objects[f"obstacle_{i}"] = obs

        # 灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply actions to the robot."""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Set joint velocity targets."""
        dof_names = self.cfg.dof_names
        self.scene["robot"].set_joint_velocity_target(
            self.actions, joint_ids=SceneEntityCfg("robot", joint_names=dof_names).resolve(self)
        )

    def _get_observations(self) -> dict:
        """Compute observations for the policy."""
        robot_pos = self.scene["robot"].data.root_pos_w[:, :2]  # (num_envs, 2)
        forwards = self._get_robot_forwards()  # (num_envs, 2)

        # 到当前目标点的向量
        target_vector = self._target_pos - robot_pos
        distance = torch.norm(target_vector, dim=-1)

        # 目标方向单位向量
        target_direction = target_vector / (distance.unsqueeze(-1) + 1e-6)

        # 观察量：[dot, cross, distance]
        dot = torch.sum(forwards * target_direction, dim=-1, keepdim=True)
        cross = (
            forwards[:, 0] * target_direction[:, 1] -
            forwards[:, 1] * target_direction[:, 0]
        ).unsqueeze(-1)
        distance_obs = distance.unsqueeze(-1)

        obs = torch.hstack((dot, cross, distance_obs))
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards. The ManagerBasedRLEnv framework
        will call individual reward functions from env_cfg and sum them."""
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions."""
        return (
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )

    def _reset_idx(self, env_ids: Optional[Sequence[int]]):
        """Reset environments with new waypoints and obstacles."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        else:
            env_ids = list(env_ids)

        if len(env_ids) == 0:
            return

        super()._reset_idx(env_ids)

        # 重置机器人位置
        default_root_state = self.scene["robot"].data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.scene["robot"].write_root_state_to_sim(default_root_state, env_ids)

        # 为每个环境生成随机障碍物
        for env_id in env_ids:
            self._generate_random_obstacles(env_id)

        # 为每个环境生成路径点
        for env_id in env_ids:
            start_pos = self.scene["robot"].data.root_pos_w[env_id, :2]
            goal_pos = self._generate_random_goal(start_pos)
            waypoints = self._plan_path(start_pos, goal_pos, env_id)
            self._waypoints[env_id] = waypoints

        # 初始化当前目标点为第一个路径点
        for env_id in env_ids:
            self._current_waypoint_idx[env_id] = 0
            self._target_pos[env_id] = self._waypoints[env_id, 0]

        # 重置距离追踪
        robot_pos = self.scene["robot"].data.root_pos_w[env_ids, :2]
        self._prev_distance[env_ids] = torch.norm(
            robot_pos - self._target_pos[env_ids], dim=-1
        )

    # ========== 辅助方法 ==========

    def _get_robot_forwards(self) -> torch.Tensor:
        """Get the forward direction of the robot in world frame."""
        quat = self.scene["robot"].data.root_link_quat_w
        forward_vec = self.scene["robot"].data.FORWARD_VEC_B
        return quat_apply(quat, forward_vec)[:, :2]

    def _generate_random_obstacles(self, env_id: int):
        """Generate random obstacle positions for one environment."""
        for obs_idx in range(self.cfg.num_obstacles):
            angle = sample_uniform(-math.pi, math.pi, (1,), self.device).item()
            distance = sample_uniform(0.5, 1.5, (1,), self.device).item()

            x = distance * math.cos(angle)
            y = distance * math.sin(angle)

            self._obstacle_positions[env_id, obs_idx] = torch.tensor(
                [x, y], device=self.device
            )

            env_origin = self.scene.env_origins[env_id, :2]
            obstacle_key = f"obstacle_{obs_idx}"
            if obstacle_key in self.scene.rigid_objects:
                obstacle = self.scene[obstacle_key]
                obstacle.set_world_poses(
                    positions=torch.tensor(
                        [[x + env_origin[0].item(), y + env_origin[1].item(), self.cfg.obstacle_height / 2]],
                        device=self.device,
                    ),
                    env_ids=torch.tensor([env_id], device=self.device),
                )

    def _generate_random_goal(self, start_pos: torch.Tensor) -> torch.Tensor:
        """Generate a random goal position away from start."""
        angle = sample_uniform(-math.pi, math.pi, (1,), self.device).item()
        distance = sample_uniform(2.0, 4.0, (1,), self.device).item()

        goal_x = start_pos[0].item() + distance * math.cos(angle)
        goal_y = start_pos[1].item() + distance * math.sin(angle)

        return torch.tensor([goal_x, goal_y], device=self.device)

    def _plan_path(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        env_id: int,
    ) -> torch.Tensor:
        """Simple waypoint planner: linear interpolation with obstacle avoidance.

        This is a placeholder for the actual Fast-Planner-like algorithm.
        For now, it generates intermediate waypoints along the path,
        avoiding obstacles by adding perpendicular offsets.

        Returns:
            Waypoints tensor of shape (num_waypoints, 2)
        """
        num_waypoints = self.cfg.num_waypoints
        waypoints = torch.zeros((num_waypoints, 2), device=self.device)

        for i in range(num_waypoints):
            t = (i + 1) / (num_waypoints + 1)
            point = start * (1 - t) + goal * t

            for obs_idx in range(self.cfg.num_obstacles):
                obs_pos = self._obstacle_positions[env_id, obs_idx]
                dist = torch.norm(point - obs_pos)

                if dist < self.cfg.obstacle_radius + 0.3:
                    direction = point - obs_pos
                    direction = direction / (dist + 1e-6)
                    point = obs_pos + direction * (self.cfg.obstacle_radius + 0.5)

            waypoints[i] = point

        return waypoints

    def _check_and_advance_waypoint(self):
        """Check if current waypoint is reached, advance to next if so."""
        robot_pos = self.scene["robot"].data.root_pos_w[:, :2]
        distance = torch.norm(robot_pos - self._target_pos, dim=-1)

        reached = distance < self.cfg.target_reach_threshold

        for env_id in reached.nonzero(as_tuple=False).squeeze(-1):
            env_id_int = int(env_id.item())
            self._current_waypoint_idx[env_id_int] += 1

            if self._current_waypoint_idx[env_id_int] < self._num_waypoints:
                self._target_pos[env_id_int] = self._waypoints[
                    env_id_int, self._current_waypoint_idx[env_id_int]
                ]
                self._prev_distance[env_id_int] = distance[env_id_int]