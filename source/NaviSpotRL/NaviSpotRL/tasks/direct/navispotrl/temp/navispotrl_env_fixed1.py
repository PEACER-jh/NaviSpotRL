# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import random
import omni.usd
import omni.kit.commands
from typing import Optional
from collections.abc import Sequence
from pxr import UsdGeom, UsdShade, Gf, Vt, Sdf

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils


def define_markers() -> VisualizationMarkers:
    """定义可视化标记：青色=机器人朝向，橙色=目标方向，红色球=目标点"""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "forward": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "target": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
            ),
            "target_point": sim_utils.SphereCfg(
                radius=0.15,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


from .navispotrl_env_cfg import NavispotrlEnvCfg

class NavispotrlEnv(DirectRLEnv):
    cfg: NavispotrlEnvCfg

    def __init__(self, cfg: NavispotrlEnvCfg, render_mode: Optional[str] = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        self._arena_centers = torch.zeros((self.num_envs, 2), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # ========== 创建围墙 ==========
        half = self.cfg.arena_side_length / 2.0
        wall_h = self.cfg.wall_height
        wall_t = self.cfg.wall_thickness
        wall_len = 2.0 * half + wall_t
        wall_configs = [
            ("/World/envs/env_.*/Wall_Right",  (half, 0.0, wall_h/2),  (wall_t, wall_len, wall_h)),
            ("/World/envs/env_.*/Wall_Left",   (-half, 0.0, wall_h/2), (wall_t, wall_len, wall_h)),
            ("/World/envs/env_.*/Wall_Front",  (0.0, half, wall_h/2),  (wall_len, wall_t, wall_h)),
            ("/World/envs/env_.*/Wall_Back",   (0.0, -half, wall_h/2), (wall_len, wall_t, wall_h)),
        ]
        for prim_path, position, size in wall_configs:
            wall_cfg = sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            )
            wall_cfg.func(prim_path, wall_cfg, translation=position)

        # ========== 创建障碍物 ==========
        margin = self.cfg.wall_margin
        if not hasattr(self, '_obstacle_data'):
            self._obstacle_data = {}
        self._obstacle_data["template"] = []

        for i in range(self.cfg.num_obstacles):
            sx = random.uniform(self.cfg.box_min_size, self.cfg.box_max_size)
            sy = random.uniform(self.cfg.box_min_size, self.cfg.box_max_size)
            sh = random.uniform(self.cfg.box_min_height, self.cfg.box_max_height)  
            for _ in range(100):
                ox = random.uniform(-half + margin, half - margin)
                oy = random.uniform(-half + margin, half - margin)
                overlap = False
                for _, oox, ooy, osx, osy, _ in self._obstacle_data["template"]:
                    if abs(ox - oox) < (sx + osx) / 2 and abs(oy - ooy) < (sy + osy) / 2:
                        overlap = True
                        break
                if not overlap:
                    break
            
            self._obstacle_data["template"].append(("box", ox, oy, sx, sy, sh)) 
            obs_cfg = sim_utils.CuboidCfg(
                size=(sx, sy, sh),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            )
            obs_cfg.func(
                f"/World/envs/env_.*/Obstacle_{i:02d}",
                obs_cfg,
                translation=(ox, oy, sh / 2),
            )

        # ========== 克隆环境 ==========
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()
        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.target_pos = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)
        self.target_yaws = torch.zeros((self.cfg.scene.num_envs, 1), device=self.device)
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)
        self.marker_offset[:, -1] = 0.5
        self.target_point_offset = torch.zeros((self.cfg.scene.num_envs, 3), device=self.device)
        self.target_point_offset[:, -1] = 0.2

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._visualize_markers()

    def _apply_action(self) -> None:
        scaled_actions = self.actions * self.cfg.action_scale
        self.robot.set_joint_velocity_target(scaled_actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        self.root_pos = self.robot.data.root_pos_w[:, :2]
        self.forwards = math_utils.quat_apply(
            self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B
        )[:, :2]

        target_vector = self.target_pos[:, :2] - self.root_pos
        self.distance_to_target = torch.norm(target_vector, dim=-1)
        target_direction = target_vector / (self.distance_to_target.unsqueeze(-1) + 1e-6)

        dot = torch.sum(self.forwards * target_direction, dim=-1, keepdim=True)
        cross = (self.forwards[:, 0] * target_direction[:, 1] -
                 self.forwards[:, 1] * target_direction[:, 0]).unsqueeze(-1)
        distance_obs = self.distance_to_target.unsqueeze(-1)

        obs = torch.hstack((dot, cross, distance_obs))
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        distance = self.distance_to_target
        if not hasattr(self, '_prev_distance'):
            self._prev_distance = distance.clone()
        distance_change = self._prev_distance - distance
        self._prev_distance = distance.clone()

        forward_speed = self.robot.data.root_com_lin_vel_b[:, 0]
        target_vector = self.target_pos[:, :2] - self.root_pos
        target_direction = target_vector / (distance.unsqueeze(-1) + 1e-6)
        alignment = torch.sum(self.forwards * target_direction, dim=-1)

        progress_reward = 5.0 * distance_change
        arrived_mask = (distance < self.cfg.target_reach_threshold).float()
        reach_bonus = 10.0 * arrived_mask
        alignment_reward = 0.5 * alignment * (1.0 - arrived_mask)
        speed_reward = 1.0 * forward_speed * (alignment > 0.9).float() * (1.0 - arrived_mask)
        
        stillness_bonus = 3.0 * (1.0 - torch.abs(forward_speed)) * arrived_mask
        leave_penalty = -10.0 * (distance - self.cfg.target_reach_threshold).clamp(min=0) * arrived_mask

        zero_action_bonus = 5.0 * (torch.abs(self.actions).sum(dim=-1) < 0.1).float() * arrived_mask

        total_reward = (progress_reward + reach_bonus + alignment_reward + 
                        speed_reward + stillness_bonus + leave_penalty + zero_action_bonus)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        reached = self.distance_to_target < self.cfg.target_reach_threshold
        return reached, time_out

    def _reset_idx(self, env_ids: Optional[Sequence[int]]):
        if env_ids is None:
            _env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            _env_ids = env_ids
        else:
            _env_ids = torch.tensor(list(env_ids), dtype=torch.long, device=self.device)

        if len(_env_ids) == 0:
            return

        super()._reset_idx(_env_ids)

        half = self.cfg.arena_side_length / 2.0
        margin = self.cfg.wall_margin

        # ========== 生成随机障碍物 ==========
        self._generate_random_obstacles(_env_ids, half, margin)

        # ========== 围墙内随机初始化 Jetbot ==========
        jetbot_positions = self._generate_random_jetbot_position(_env_ids, half, margin)

        default_root_state = self.robot.data.default_root_state[_env_ids]
        default_root_state[:, :3] += self.scene.env_origins[_env_ids]
        default_root_state[:, 0] += jetbot_positions[:, 0]
        default_root_state[:, 1] += jetbot_positions[:, 1]

        rand_yaw = sample_uniform(-math.pi, math.pi, (len(_env_ids), 1), self.device).squeeze(-1)
        default_root_state[:, 3] = torch.cos(rand_yaw / 2)
        default_root_state[:, 6] = torch.sin(rand_yaw / 2)
        self.robot.write_root_state_to_sim(default_root_state, _env_ids)

        # ========== 在围墙内随机生成目标点 ==========
        robot_xy = self.robot.data.root_pos_w[_env_ids, :2]
        env_origin_xy = self.scene.env_origins[_env_ids, :2]
        target_positions = self._generate_random_target_position(_env_ids, robot_xy, env_origin_xy, half, margin)

        self.target_pos[_env_ids, 0] = target_positions[:, 0]
        self.target_pos[_env_ids, 1] = target_positions[:, 1]
        self.target_pos[_env_ids, 2] = 0.0

        # 更新可视化
        target_vector = self.target_pos[_env_ids, :2] - robot_xy
        self.target_yaws[_env_ids] = torch.atan2(target_vector[:, 1], target_vector[:, 0]).unsqueeze(-1)

        if hasattr(self, '_prev_distance'):
            self._prev_distance[_env_ids] = torch.norm(self.target_pos[_env_ids, :2] - robot_xy, dim=-1)

        self._visualize_markers()

    def _visualize_markers(self):
        robot_pos = self.robot.data.root_pos_w
        forward_quat = self.robot.data.root_quat_w
        target_quat = math_utils.quat_from_angle_axis(self.target_yaws, self.up_dir).squeeze()

        forward_loc = robot_pos + self.marker_offset
        target_loc = robot_pos + self.marker_offset
        target_point_loc = self.target_pos + self.target_point_offset
        target_point_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        loc = torch.vstack((forward_loc, target_loc, target_point_loc))
        rots = torch.vstack((forward_quat, target_quat, target_point_rot))

        all_envs = torch.arange(self.cfg.scene.num_envs, device=self.device)
        indices = torch.hstack((
            torch.zeros_like(all_envs),
            torch.ones_like(all_envs),
            2 * torch.ones_like(all_envs),
        ))

        self.visualization_markers.visualize(loc, rots, marker_indices=indices)
    
    def _generate_random_obstacles(self, env_ids: torch.Tensor, half: float, margin: float):
        if not hasattr(self, '_obstacle_data'):
            self._obstacle_data = {}

        for env_id in env_ids.tolist():
            self._obstacle_data[env_id] = []

            for obs_idx in range(self.cfg.num_obstacles):
                prim_path = f"/World/envs/env_{env_id}/Obstacle_{obs_idx:02d}"
                stage = omni.usd.get_context().get_stage()
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue

                sx = random.uniform(self.cfg.box_min_size, self.cfg.box_max_size)
                sy = random.uniform(self.cfg.box_min_size, self.cfg.box_max_size)
                sh = random.uniform(self.cfg.box_min_height, self.cfg.box_max_height)
                for _ in range(100):
                    ox = random.uniform(-half + margin, half - margin)
                    oy = random.uniform(-half + margin, half - margin)
                    if not self._is_inside_any_obstacle(ox, oy, env_id, safe_margin=0.0):
                        self._obstacle_data[env_id].append(("box", ox, oy, sx, sy, sh))
                        break
                else:
                    ox = random.uniform(-half + margin, half - margin)
                    oy = random.uniform(-half + margin, half - margin)
                    self._obstacle_data[env_id].append(("box", ox, oy, sx, sy, sh))

                translate_attr = prim.GetAttribute("xformOp:translate")
                scale_attr = prim.GetAttribute("xformOp:scale")

                if translate_attr.IsValid():
                    translate_attr.Set(Gf.Vec3d(ox, oy, sh / 2))
                if scale_attr.IsValid():
                    scale_attr.Set(Gf.Vec3d(sx, sy, sh))

    def _generate_random_jetbot_position(
        self, env_ids: torch.Tensor, half: float, margin: float
    ) -> torch.Tensor:
        positions = torch.zeros((len(env_ids), 2), device=self.device)
        for i, env_id in enumerate(env_ids.tolist()):
            for _ in range(100):
                lx = random.uniform(-half + margin, half - margin)
                ly = random.uniform(-half + margin, half - margin)
                if not self._is_inside_any_obstacle(lx, ly, env_id, self.cfg.obstacle_safe_margin):
                    positions[i, 0] = lx
                    positions[i, 1] = ly
                    break
        return positions

    def _generate_random_target_position(
        self, env_ids: torch.Tensor, robot_xy: torch.Tensor, 
        env_origin_xy: torch.Tensor, half: float, margin: float
    ) -> torch.Tensor:
        target_positions = torch.zeros((len(env_ids), 2), device=self.device)
        target_margin = margin + 0.5

        for i, env_id in enumerate(env_ids.tolist()):
            env_ox = env_origin_xy[i, 0].item()
            env_oy = env_origin_xy[i, 1].item()
            robot_x = robot_xy[i, 0].item()
            robot_y = robot_xy[i, 1].item()

            for _ in range(100):
                dist = random.uniform(self.cfg.target_distance_range[0], self.cfg.target_distance_range[1])
                angle = random.uniform(-math.pi, math.pi)
                wx = robot_x + dist * math.cos(angle)
                wy = robot_y + dist * math.sin(angle)

                lx = wx - env_ox
                ly = wy - env_oy
                lx = max(-half + target_margin, min(half - target_margin, lx))
                ly = max(-half + target_margin, min(half - target_margin, ly))

                if not self._is_inside_any_obstacle(lx, ly, env_id, self.cfg.obstacle_safe_margin):
                    target_positions[i, 0] = env_ox + lx
                    target_positions[i, 1] = env_oy + ly
                    break
            else:
                for _ in range(100):
                    lx = random.uniform(-half + target_margin, half - target_margin)
                    ly = random.uniform(-half + target_margin, half - target_margin)
                    if not self._is_inside_any_obstacle(lx, ly, env_id, self.cfg.obstacle_safe_margin):
                        target_positions[i, 0] = env_ox + lx
                        target_positions[i, 1] = env_oy + ly
                        break

        return target_positions

    def _is_inside_any_obstacle(
        self, x: float, y: float, env_id: int, safe_margin: float = 0.15
    ) -> bool:
        if env_id not in self._obstacle_data:
            return False
        for obs in self._obstacle_data[env_id]:
            _, ox, oy, sx, sy, _ = obs
            half_sx = sx / 2 + safe_margin
            half_sy = sy / 2 + safe_margin
            if abs(x - ox) < half_sx and abs(y - oy) < half_sy:
                return True

        return False
