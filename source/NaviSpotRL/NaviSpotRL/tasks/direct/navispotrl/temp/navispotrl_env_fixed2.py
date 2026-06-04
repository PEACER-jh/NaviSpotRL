# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Claus

'''
python ./scripts/rsl_rl/train.py --task=Template-Navispotrl-Direct-v0 \
                                --num_envs 64 --enable_cameras --max_iterations 500

pkill -9 -f train.py; pkill -9 -f isaac.sim; sleep 2; echo "cleaned"
'''

from __future__ import annotations

import os
import cv2
import uuid
import math
import torch
import shutil
import atexit
import random
import rerun as rr
import numpy as np
import omni.usd
import omni.physx
import omni.kit.app
import omni.kit.commands
import omni.physx.bindings._physx as pb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.simulation_manager import SimulationManager
from typing import Optional
from collections.abc import Sequence
from torch.utils.tensorboard import SummaryWriter
from pxr import UsdGeom, UsdUtils, UsdPhysics, PhysxSchema, Gf, Sdf, Usd

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import sample_uniform
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
        # if os.path.exists(cfg.tb_log_dir):
        #     shutil.rmtree(cfg.tb_log_dir)
        # os.makedirs(cfg.tb_log_dir, exist_ok=True)
        # self._log_counter = 0
        # self.writer = SummaryWriter(log_dir=cfg.tb_log_dir)

        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        self._obstacle_data = {"template": []}

        # rr.init("navi_camera", spawn=True)
        # rr.set_time_sequence("step", 0)
        # self._rerun_step = 0

    def _setup_scene(self):
        if not hasattr(self, '_obstacle_data'):
            self._obstacle_data = {"template": []}
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        half = self.cfg.arena_side_length / 2.0
        wall_h = self.cfg.wall_height
        wall_t = self.cfg.wall_thickness
        wall_len = 2.0 * half + wall_t
        wall_cfgs = [
            ("/World/envs/env_.*/Wall_Right",  (half, 0.0, wall_h/2),  (wall_t, wall_len, wall_h)),
            ("/World/envs/env_.*/Wall_Left",   (-half, 0.0, wall_h/2), (wall_t, wall_len, wall_h)),
            ("/World/envs/env_.*/Wall_Front",  (0.0, half, wall_h/2),  (wall_len, wall_t, wall_h)),
            ("/World/envs/env_.*/Wall_Back",   (0.0, -half, wall_h/2), (wall_len, wall_t, wall_h)),
        ]
        for prim_path, pos, size in wall_cfgs:
            cfg = sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            )
            cfg.func(prim_path, cfg, translation=pos)

        # ========== 创建障碍物 ==========
        margin = self.cfg.wall_safe_margin
        temp_obs = []
        for i in range(self.cfg.num_obstacles):
            sx = random.uniform(self.cfg.box_min_size, self.cfg.box_max_size)
            sy = random.uniform(self.cfg.box_min_size, self.cfg.box_max_size)
            sh = random.uniform(self.cfg.box_min_height, self.cfg.box_max_height)
            
            for _ in range(200):
                ox = random.uniform(-half + margin, half - margin)
                oy = random.uniform(-half + margin, half - margin)
                overlap = False
                for oox, ooy, osx, osy in temp_obs:
                    if abs(ox - oox) < (sx + osx) / 2 and abs(oy - ooy) < (sy + osy) / 2:
                        overlap = True
                        break
                if not overlap:
                    break
            
            temp_obs.append((ox, oy, sx, sy))
            obs_cfg = sim_utils.CuboidCfg(
                size=(sx, sy, sh),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=True,
                    linear_damping=0.0,
                    angular_damping=0.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1e9),
            )
            obs_cfg.func(
                f"/World/envs/env_.*/Obstacle_{i:02d}",
                obs_cfg,
                translation=(ox, oy, sh / 2),
            )

        # ========== 注册机器人 ==========
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        stage = omni.usd.get_context().get_stage()

        # ========== 创建相机 ==========
        # 前视相机
        camera_front_cfg = CameraCfg(
            prim_path="/World/envs/env_.*/Robot/panoramic_rover/base_link/camera_front_link/CameraFront",
            update_period=0.016,
            height=240, width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 30.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0),
                convention="world",
            ),
        )
        self._camera_front = Camera(cfg=camera_front_cfg)

        # 后视相机
        camera_back_cfg = CameraCfg(
            prim_path="/World/envs/env_.*/Robot/panoramic_rover/base_link/camera_back_link/CameraBack",
            update_period=0.016,
            height=240, width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 30.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0),
                convention="world",
            ),
        )
        self._camera_back = Camera(cfg=camera_back_cfg)

        # ========== 克隆环境 ==========
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()
        self.up_dir = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        # Initialize target at visible default (env origin + offset) so markers never disappear
        self.target_pos = self.scene.env_origins[:, :3] + torch.tensor([0.0, 0.0, 0.2], device=self.device)
        self.target_yaws = torch.zeros((self.num_envs, 1), device=self.device)
        self.marker_offset = torch.zeros((self.num_envs, 3), device=self.device)
        self.marker_offset[:, -1] = 0.5
        self.target_point_offset = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_point_offset[:, -1] = 0.2

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        if not self.sim.is_fabric_enabled():
            self._sync_robot_pose_to_usd()

        self._visualize_markers()
        self._camera_front.update(self.step_dt)
        self._camera_back.update(self.step_dt)

    def _sync_robot_pose_to_usd(self):
        stage = omni.usd.get_context().get_stage()
        body_pose_w = self.robot.data.body_link_pose_w.detach().cpu().numpy()
        num_bodies = len(self.robot.body_names)

        with Sdf.ChangeBlock():
            for env_id in range(self.num_envs):
                env_root_prim_path = f"/World/envs/env_{env_id}/Robot"
                root_prim = stage.GetPrimAtPath(env_root_prim_path)
                if not root_prim.IsValid():
                    continue

                rwx = float(body_pose_w[env_id, 0, 0])
                rwy = float(body_pose_w[env_id, 0, 1])
                rwz = float(body_pose_w[env_id, 0, 2])
                rqw = float(body_pose_w[env_id, 0, 3])
                rqx = float(body_pose_w[env_id, 0, 4])
                rqy = float(body_pose_w[env_id, 0, 5])
                rqz = float(body_pose_w[env_id, 0, 6])
                root_world_mat = Gf.Matrix4d()
                root_world_mat.SetTranslateOnly(Gf.Vec3d(rwx, rwy, rwz))
                root_world_mat.SetRotateOnly(Gf.Quatd(rqw, rqx, rqy, rqz))
                root_world_inv = root_world_mat.GetInverse()

                for body_idx in range(num_bodies):
                    body_name = self.robot.body_names[body_idx]
                    if body_idx == 0:
                        prim = root_prim
                    else:
                        prim = stage.GetPrimAtPath(f"{env_root_prim_path}/{body_name}")
                        if not prim.IsValid():
                            continue

                    wx = float(body_pose_w[env_id, body_idx, 0])
                    wy = float(body_pose_w[env_id, body_idx, 1])
                    wz = float(body_pose_w[env_id, body_idx, 2])
                    qw = float(body_pose_w[env_id, body_idx, 3])
                    qx = float(body_pose_w[env_id, body_idx, 4])
                    qy = float(body_pose_w[env_id, body_idx, 5])
                    qz = float(body_pose_w[env_id, body_idx, 6])
                    body_world_mat = Gf.Matrix4d()
                    body_world_mat.SetTranslateOnly(Gf.Vec3d(wx, wy, wz))
                    body_world_mat.SetRotateOnly(Gf.Quatd(qw, qx, qy, qz))

                    if body_idx == 0:
                        parent = prim.GetParent()
                        if parent.IsValid() and parent.GetPath() != Sdf.Path.absoluteRootPath:
                            xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                            parent_world = xf_cache.GetLocalToWorldTransform(parent)
                            local_mat = body_world_mat * parent_world.GetInverse()
                            lx = local_mat.ExtractTranslation()[0]
                            ly = local_mat.ExtractTranslation()[1]
                            lz = local_mat.ExtractTranslation()[2]
                            local_quat = local_mat.ExtractRotationQuat()
                            lqw = local_quat.GetReal()
                            lqv = local_quat.GetImaginary()
                            lqx = lqv[0]; lqy = lqv[1]; lqz = lqv[2]
                        else:
                            lx, ly, lz = wx, wy, wz
                            lqw, lqx, lqy, lqz = qw, qx, qy, qz
                    else:
                        local_mat = body_world_mat * root_world_inv
                        lx = local_mat.ExtractTranslation()[0]
                        ly = local_mat.ExtractTranslation()[1]
                        lz = local_mat.ExtractTranslation()[2]
                        local_quat = local_mat.ExtractRotationQuat()
                        lqw = local_quat.GetReal()
                        lqv = local_quat.GetImaginary()
                        lqx = lqv[0]; lqy = lqv[1]; lqz = lqv[2]

                    if body_idx == 0 and lz < 0.005:
                    # (Z-CLAMP logging disabled)
                        lz = 0.04

                    try:
                        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(lx, ly, lz))
                        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(lqw, lqx, lqy, lqz))
                    except Exception:
                        pass

    def _apply_action(self) -> None:
        scaled_actions = self.actions * self.cfg.action_scale
        four_wheel_actions = torch.zeros((self.num_envs, 4), device=self.device)
        four_wheel_actions[:, 0] = scaled_actions[:, 0]  # left_wheel_joint
        four_wheel_actions[:, 1] = scaled_actions[:, 0]  # left_wheel_back_joint
        four_wheel_actions[:, 2] = scaled_actions[:, 1]  # right_wheel_joint
        four_wheel_actions[:, 3] = scaled_actions[:, 1]  # right_wheel_back_joint
        self.robot.set_joint_velocity_target(four_wheel_actions, joint_ids=self.dof_idx)

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
        # self._visualize_cameras()
        

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        distance = self.distance_to_target
        if not hasattr(self, '_prev_distance'):
            self._prev_distance = distance.clone()
        distance_change = self._prev_distance - distance
        self._prev_distance = distance.clone()

        target_vector = self.target_pos[:, :2] - self.root_pos
        target_direction = target_vector / (distance.unsqueeze(-1) + 1e-6)
        alignment = torch.sum(self.forwards * target_direction, dim=-1)
        forward_vel = self.robot.data.root_com_lin_vel_b[:, 0]
        ang_vel = self.robot.data.root_com_ang_vel_b[:, 2]
        arrived = (distance < self.cfg.target_reach_threshold).float()
        align_reward = alignment
        fwd = torch.tanh(forward_vel * 0.5)
        speed_reward = fwd * alignment.clamp(min=0.0)
        spin_penalty = -(ang_vel.abs()) * (alignment > 0.7).float()
        progress_reward = distance_change * 2.0
        arrival_bonus = arrived * 20.0
        time_penalty = -0.02 * (1.0 - arrived)

        total_reward = (
            0.8 * align_reward
            + 1.5 * speed_reward
            + 1.0 * spin_penalty
            + progress_reward
            + arrival_bonus
            + time_penalty
        )

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

        half = self.cfg.arena_side_length / 2.0
        margin = self.cfg.wall_safe_margin

        super()._reset_idx(_env_ids)
        
        self._update_obstacles(_env_ids, half, margin)

        # Generate robot position
        robot_positions = self._generate_random_robot_position(_env_ids, half, margin)
        default_root_state = self.robot.data.default_root_state[_env_ids].clone()
        env_origins = self.scene.env_origins[_env_ids, :3]
        init_z = self.cfg.robot_cfg.init_state.pos[2]
        default_root_state[:, 0] = env_origins[:, 0] + robot_positions[:, 0]
        default_root_state[:, 1] = env_origins[:, 1] + robot_positions[:, 1]
        default_root_state[:, 2] = env_origins[:, 2] + init_z
        rand_yaw = sample_uniform(-math.pi, math.pi, (len(_env_ids), 1), self.device).squeeze(-1)
        default_root_state[:, 3] = torch.cos(rand_yaw / 2)
        default_root_state[:, 4] = 0.0
        default_root_state[:, 5] = 0.0
        default_root_state[:, 6] = torch.sin(rand_yaw / 2)

        # Update robot USD prim for rendering (non-fabric only)
        if not self.sim.is_fabric_enabled():
            stage = omni.usd.get_context().get_stage()
            for i, env_id in enumerate(_env_ids.tolist()):
                prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot")
                if prim.IsValid():
                    wx = default_root_state[i, 0].item()
                    wy = default_root_state[i, 1].item()
                    wz = default_root_state[i, 2].item()
                    qw = default_root_state[i, 3].item()
                    qx = default_root_state[i, 4].item()
                    qy = default_root_state[i, 5].item()
                    qz = default_root_state[i, 6].item()
                    parent = prim.GetParent()
                    if parent.IsValid() and parent.GetPath() != Sdf.Path.absoluteRootPath:
                        xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                        world_mat = Gf.Matrix4d()
                        world_mat.SetTranslateOnly(Gf.Vec3d(wx, wy, wz))
                        world_mat.SetRotateOnly(Gf.Quatd(qw, qx, qy, qz))
                        parent_world = xf_cache.GetLocalToWorldTransform(parent)
                        local_mat = world_mat * parent_world.GetInverse()
                        local_trans = local_mat.ExtractTranslation()
                        local_quat = local_mat.ExtractRotationQuat()
                        lz_safe = max(float(local_trans[2]), float(init_z))
                        prim.GetAttribute("xformOp:translate").Set(
                            Gf.Vec3d(local_trans[0], local_trans[1], lz_safe))
                        prim.GetAttribute("xformOp:orient").Set(local_quat)
                    else:
                        wz_safe = max(wz, float(init_z))
                        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(wx, wy, wz_safe))
                        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(qw, qx, qy, qz))

        # Write robot root state
        self.robot.write_root_state_to_sim(default_root_state, _env_ids)

        # Generate target
        robot_xy = self.robot.data.root_pos_w[_env_ids, :2]
        env_origin_xy = self.scene.env_origins[_env_ids, :2]
        target_positions = self._generate_random_target_position(_env_ids, robot_xy, env_origin_xy, half, margin)
        self.target_pos[_env_ids, 0] = target_positions[:, 0]
        self.target_pos[_env_ids, 1] = target_positions[:, 1]
        self.target_pos[_env_ids, 2] = 0.0
        target_vector = self.target_pos[_env_ids, :2] - robot_xy
        self.target_yaws[_env_ids] = torch.atan2(target_vector[:, 1], target_vector[:, 0]).unsqueeze(-1)
        if hasattr(self, '_prev_distance'):
            self._prev_distance[_env_ids] = torch.norm(self.target_pos[_env_ids, :2] - robot_xy, dim=-1)
        self._visualize_markers(robot_world_pos=default_root_state[:, :3])
    def _visualize_markers(self, robot_world_pos: torch.Tensor | None = None):
        robot_pos = robot_world_pos if robot_world_pos is not None else self.robot.data.root_pos_w
        forward_quat = self.robot.data.root_quat_w
        target_quat = math_utils.quat_from_angle_axis(self.target_yaws, self.up_dir).squeeze()

        forward_loc = robot_pos + self.marker_offset
        target_loc = robot_pos + self.marker_offset
        target_point_loc = self.target_pos + self.target_point_offset
        target_point_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # Ensure target_quat has correct shape (num_envs, 4) even after squeeze
        if target_quat.dim() == 1:
            target_quat = target_quat.unsqueeze(0)

        loc = torch.vstack((forward_loc, target_loc, target_point_loc))
        rots = torch.vstack((forward_quat, target_quat, target_point_rot))

        all_envs = torch.arange(self.cfg.scene.num_envs, device=self.device)
        indices = torch.hstack((
            torch.zeros_like(all_envs),
            torch.ones_like(all_envs),
            2 * torch.ones_like(all_envs),
        ))
        indices = indices.to(dtype=torch.int32)

        self.visualization_markers.visualize(loc, rots, marker_indices=indices)

    def _visualize_cameras(self):
        # # ========== TensorBoard 可视化所有环境 ==========
        # front_rgb = self._camera_front.data.output["rgb"][0].permute(2, 0, 1)
        # back_rgb = self._camera_back.data.output["rgb"][0].permute(2, 0, 1)

        # self._log_counter += 1
        # step = self._log_counter % 10
        # self.writer.add_image("camera/front", front_rgb, global_step=step)
        # self.writer.add_image("camera/back", back_rgb, global_step=step)
        # self.writer.flush()
        # ========== Rerun 实时显示 ==========
        front_rgb = self._camera_front.data.output["rgb"]
        front_depth = self._camera_front.data.output.get("distance_to_image_plane")
        back_rgb = self._camera_back.data.output["rgb"]
        back_depth = self._camera_back.data.output.get("distance_to_image_plane")

        front_rgb_np = front_rgb[0].cpu().numpy()
        front_depth_np = front_depth[0].squeeze(-1).cpu().numpy()
        front_depth_np = np.clip(front_depth_np, 0, 15.0)
        front_depth_color = cv2.applyColorMap((front_depth_np / 15.0 * 255).astype(np.uint8), cv2.COLORMAP_JET)
        front_combined = np.vstack([front_rgb_np, front_depth_color])

        back_rgb_np = back_rgb[0].cpu().numpy()
        back_depth_np = back_depth[0].squeeze(-1).cpu().numpy()
        back_depth_np = np.clip(back_depth_np, 0, 15.0)
        back_depth_color = cv2.applyColorMap((back_depth_np / 15.0 * 255).astype(np.uint8), cv2.COLORMAP_JET)
        back_combined = np.vstack([back_rgb_np, back_depth_color])

        rr.set_time_sequence("step", self._rerun_step)
        rr.log("camera/front", rr.Image(front_combined))
        rr.log("camera/back", rr.Image(back_combined))
        self._rerun_step += 1

    # ===================== 障碍物更新 =====================

    def _update_obstacles(self, env_ids: torch.Tensor, half: float, margin: float):
        """Delete old obstacle prims, recreate with new params, then reload physics."""
        stage = omni.usd.get_context().get_stage()
        env_origins = self.scene.env_origins
        obs_params = []

        for env_id in env_ids.tolist():
            self._obstacle_data[env_id] = []
            for i in range(self.cfg.num_obstacles):
                prim_path = f"/World/envs/env_{env_id}/Obstacle_{i:02d}"
                sx = random.uniform(self.cfg.box_min_size, self.cfg.box_max_size)
                sy = random.uniform(self.cfg.box_min_size, self.cfg.box_max_size)
                sh = random.uniform(self.cfg.box_min_height, self.cfg.box_max_height)
                ox = 0.0
                oy = 0.0

                placed = False
                for reduce in [1.0, 0.85, 0.7, 0.55, 0.4]:
                    _sx = sx * reduce
                    _sy = sy * reduce
                    for _ in range(200):
                        ox = random.uniform(-half + margin, half - margin)
                        oy = random.uniform(-half + margin, half - margin)
                        if not self._check_overlap(env_id, ox, oy, _sx, _sy):
                            sx, sy = _sx, _sy
                            placed = True
                            break
                    if placed:
                        break

                if reduce < 1.0:
                    sh *= reduce

                self._obstacle_data[env_id].append(("box", ox, oy, sx, sy, sh))
                obs_params.append((env_id, i, prim_path, ox, oy, sx, sy, sh))

        for _, _, prim_path, _, _, _, _, _ in obs_params:
            old_prim = stage.GetPrimAtPath(prim_path)
            if old_prim.IsValid():
                stage.RemovePrim(prim_path)

        for _, _, prim_path, ox, oy, sx, sy, sh in obs_params:
            obs_cfg = sim_utils.CuboidCfg(
                size=(sx, sy, sh),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                ),
            )
            obs_cfg.func(
                prim_path,
                obs_cfg,
                translation=(ox, oy, sh / 2),
            )

        omni.physx.acquire_physx_interface().force_load_physics_from_usd()
        self.robot._initialize_impl()

    def _generate_random_robot_position(
        self, env_ids: torch.Tensor, half: float, margin: float
    ) -> torch.Tensor:
        """Generate robot positions, avoiding obstacles with safe_margin."""
        positions = torch.zeros((len(env_ids), 2), device=self.device)
        safe_margin = self.cfg.obstacle_safe_margin
        arena_min = -half + margin
        arena_max = half - margin

        for i, env_id in enumerate(env_ids.tolist()):
            best_lx, best_ly = 0.0, 0.0
            best_clearance = -1.0

            for _ in range(2000):
                lx = random.uniform(arena_min, arena_max)
                ly = random.uniform(arena_min, arena_max)
                if not self._is_inside_any_obstacle(lx, ly, env_id, safe_margin):
                    clearance = self._dist_to_nearest_obstacle(lx, ly, env_id)
                    if clearance > best_clearance:
                        best_clearance = clearance
                        best_lx, best_ly = lx, ly
                    if clearance > safe_margin * 2.0:
                        break

            if best_clearance >= 0.0:
                positions[i, 0] = best_lx
                positions[i, 1] = best_ly
                continue

            found = False
            for step in range(40):
                gx = arena_min + step * (arena_max - arena_min) / 39.0
                for sty in range(40):
                    gy = arena_min + sty * (arena_max - arena_min) / 39.0
                    if not self._is_inside_any_obstacle(gx, gy, env_id, safe_margin):
                        positions[i, 0] = gx
                        positions[i, 1] = gy
                        found = True
                        break
                if found:
                    break
            if found:
                continue

            for step in range(40):
                gx = arena_min + step * (arena_max - arena_min) / 39.0
                for sty in range(40):
                    gy = arena_min + sty * (arena_max - arena_min) / 39.0
                    if not self._is_inside_any_obstacle(gx, gy, env_id, safe_margin * 0.3):
                        positions[i, 0] = gx
                        positions[i, 1] = gy
                        found = True
                        break
                if found:
                    break

            if not found:
                print(f"[Warning] env={env_id} no safe robot position found! arena=[{arena_min:.1f},{arena_max:.1f}]")
                positions[i, 0] = 0.0
                positions[i, 1] = 0.0

        return positions
    
    def _generate_random_target_position(
        self, env_ids: torch.Tensor, robot_xy: torch.Tensor,
        env_origin_xy: torch.Tensor, half: float, margin: float
    ) -> torch.Tensor:
        """Generate target positions strictly within arena and outside obstacles."""
        target_positions = torch.zeros((len(env_ids), 2), device=self.device)
        target_margin = margin
        safe_margin = self.cfg.obstacle_safe_margin
        dist_min, dist_max = self.cfg.target_distance_range

        for i, env_id in enumerate(env_ids.tolist()):
            env_ox = env_origin_xy[i, 0].item()
            env_oy = env_origin_xy[i, 1].item()
            robot_x = robot_xy[i, 0].item()
            robot_y = robot_xy[i, 1].item()

            found = False
            for _ in range(500):
                dist = random.uniform(dist_min, dist_max)
                angle = random.uniform(-math.pi, math.pi)
                tx = robot_x + dist * math.cos(angle)
                ty = robot_y + dist * math.sin(angle)
                lx = tx - env_ox
                ly = ty - env_oy

                if lx < -half + target_margin or lx > half - target_margin:
                    continue
                if ly < -half + target_margin or ly > half - target_margin:
                    continue

                if not self._is_inside_any_obstacle(lx, ly, env_id, safe_margin):
                    target_positions[i, 0] = tx
                    target_positions[i, 1] = ty
                    found = True
                    break

            if not found:
                print(f"[Warning] env={env_id} target not found, using grid search")
                for step in range(20):
                    gx = -half + target_margin + step * (half - target_margin) * 2 / 19.0
                    for sty in range(20):
                        gy = -half + target_margin + sty * (half - target_margin) * 2 / 19.0
                        dist = math.hypot(gx - (robot_x - env_ox), gy - (robot_y - env_oy))
                        if dist >= dist_min * 0.8:
                            if not self._is_inside_any_obstacle(gx, gy, env_id, safe_margin * 0.5):
                                target_positions[i, 0] = env_ox + gx
                                target_positions[i, 1] = env_oy + gy
                                found = True
                                break
                    if found:
                        break
            if not found:
                print(f"[Error] env={env_id} absolutely no safe target position!")
                target_positions[i, 0] = robot_x
                target_positions[i, 1] = robot_y


        return target_positions

    # ===================== 碰撞检测 =====================

    def _dist_to_nearest_obstacle(
        self, x: float, y: float, env_id: int
    ) -> float:
        """Return distance from (x,y) to the nearest obstacle EDGE (>=0 means outside all)."""
        if env_id not in self._obstacle_data:
            return 1e9
        min_dist = 1e9
        for _, ox, oy, sx, sy, _ in self._obstacle_data[env_id]:
            dx = abs(x - ox) - sx / 2.0
            dy = abs(y - oy) - sy / 2.0
            if dx < 0 and dy < 0:
                return max(dx, dy)
            outside = max(dx, 0.0) + max(dy, 0.0)
            if outside < min_dist:
                min_dist = outside

        return min_dist

    def _is_inside_any_obstacle(
        self, x: float, y: float, env_id: int, safe_margin: float = 0.15
    ) -> bool:
        """Check if (x,y) (env-local) is inside any obstacle using _obstacle_data."""
        if env_id not in self._obstacle_data:
            return False
        for _, ox, oy, sx, sy, _ in self._obstacle_data[env_id]:
            half_sx = sx / 2.0 + safe_margin
            half_sy = sy / 2.0 + safe_margin
            if abs(x - ox) < half_sx and abs(y - oy) < half_sy:
                return True

        return False

    def _check_overlap(self, env_id: int, x: float, y: float, sx: float, sy: float) -> bool:
        if env_id not in self._obstacle_data:
            return False
        for _, ox, oy, osx, osy, _ in self._obstacle_data[env_id]:
            if abs(x - ox) < (sx + osx) * 0.5 and abs(y - oy) < (sy + osy) * 0.5:
                return True

        return False