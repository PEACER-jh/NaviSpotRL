# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Claus

'''
python ./scripts/rsl_rl/train.py --task=Template-Navispotrl-Direct-v0 \
                                --num_envs 1 --max_iterations 5000 --enable_cameras --headless

python ./scripts/rsl_rl/resume.py \
    --task Template-Navispotrl-Direct-v0 \
    --num_envs 32 --max_iterations 20000 --enable_cameras \
    --resume_path logs/rsl_rl/cartpole_direct/2026-05-28_17-02-34/model_2100.pt

python ./scripts/rsl_rl/play.py --task=Template-Navispotrl-Direct-v0 \
                                --num_envs=1 --enable_cameras

pkill -9 -f train.py; pkill -9 -f isaac.sim; sleep 2; echo "cleaned"

pkill -9 -f play.py; pkill -9 -f isaac.sim; sleep 2; echo "cleaned"

pkill -9 -f resume.py; pkill -9 -f isaac.sim; sleep 2; echo "cleaned"

rm -rf logs/rsl_rl/cartpole_direct/* outputs/*
'''

from __future__ import annotations

import os
import math
import torch
import omni.usd
import omni.physx
import rerun as rr
from typing import Optional, cast
from collections.abc import Sequence
from pxr import UsdGeom, Gf, Sdf, Usd

import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .navispotrl_env_cfg import NavispotrlEnvCfg
from .utils import scene_setup
from .utils.visualize_tools import (
    update_markers, update_path_markers, sync_robot_pose_to_usd, log_camera_views,
    init_lidar_ray_markers, update_lidar_ray_markers,
)
from .utils.global_planner import plan_global_path
from .utils.reward_func import (
    alignment_reward,
    distance_progress_reward, speed_reward, steer_toward_target_reward,
    misalign_speed_penalty, idle_penalty,
    side_drive_cost, yaw_damping, obstacle_avoid_reward, waypoint_reach,
    tilt_penalty, stuck_penalty, backward_drive_penalty,
    tilt_penalty, stuck_penalty, backward_drive_penalty,
)
from .utils.navigation import (
    waypoints_local_to_world, has_valid_path,
    project_onto_path, get_current_subgoal, get_waypoint_obs,
    check_replan_needed, decrement_cooldowns,
)
from .utils.depth_utils import depth_to_rays
from .models.debug_logger import dbg
from .utils.update_envs import (
    update_obstacles, build_occupancy_grid, grid_sample_free,
    is_free_in_grid, generate_random_robot_position,
    generate_random_target_position, dist_to_nearest_obstacle,
    is_inside_any_obstacle, check_obstacle_overlap,
)

class NavispotrlEnv(DirectRLEnv):

    def __init__(self, cfg: NavispotrlEnvCfg, render_mode: Optional[str] = None, **kwargs):
        # if os.path.exists(cfg.tb_log_dir):
        #     shutil.rmtree(cfg.tb_log_dir)
        # os.makedirs(cfg.tb_log_dir, exist_ok=True)
        # self._log_counter = 0
        # self.writer = SummaryWriter(log_dir=cfg.tb_log_dir)

        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        self._obstacle_data: dict[int, list] = {0: []}  # int key = env_id
        self._occupancy_grids = {}  # env_id -> (grid, origin_x, origin_y, resolution)
        self._global_path = None  # latest global plan waypoints

        if cfg.is_camera_log:
            rr.init("navi_camera", spawn=True)
            rr.set_time_sequence("step", 0)
            self._rerun_step = 0

    def _setup_scene(self):
        # ---- State init ----
        if not hasattr(self, '_obstacle_data'):
            self._obstacle_data: dict[int, list] = {0: []}
        self._occupancy_grids = {}

        # ---- Robot ----
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        # ---- Arena walls & obstacles (shared across envs) ----
        half = self.cfg.arena_side_length / 2.0
        scene_setup.spawn_arena_walls(half, self.cfg.wall_height, self.cfg.wall_thickness)
        scene_setup.spawn_shared_obstacles(
            self.cfg.num_obstacles, half, self.cfg.wall_safe_margin,
            self.cfg.box_min_size, self.cfg.box_max_size,
            self.cfg.box_min_height, self.cfg.box_max_height,
        )

        # ---- Clone envs & register robot ----
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        # ---- Depth cameras ----
        self._camera_front = scene_setup.create_front_depth_camera()
        self._camera_back = scene_setup.create_back_depth_camera()

        # ---- Hide all robots from depth cameras (secondary rays only) ----
        scene_setup.hide_robots_from_depth_cameras()

        # ---- Lidar ray visualization markers ----
        if self.cfg.is_ray_visualization:
            init_lidar_ray_markers(self)

        # ---- Lighting & markers ----
        scene_setup.setup_markers_and_lights(self)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        # Clear projection cache at the start of each step
        self._cached_closest_idx = None
        self._cached_cross_track = None
        # Check if any deviated robot needs path replanning
        self._try_replan()
        if not self.sim.is_fabric_enabled():
            self._sync_robot_pose_to_usd()

        if self.render_mode != "headless":
            self._visualize_markers()
        self._camera_front.update(self.step_dt)
        self._camera_back.update(self.step_dt)
        if self.render_mode != "headless":
            self._visualize_cameras()

    def _sync_robot_pose_to_usd(self):
        sync_robot_pose_to_usd(self)

    def _apply_action(self) -> None:
        # Clamp actions to [-1, 1]
        actions = torch.clamp(self.actions, -1.0, 1.0)

        # Freeze dead robots
        if hasattr(self, "_robot_dead") and self._robot_dead.any():
            actions = actions.clone()
            actions[self._robot_dead] = 0.0

        # v45: HYBRID DECOUPLED ACTION — [throttle, steering]
        #   throttle ∈ [-1, 1] → scaled to [0, max_linear_vel] (never negative!)
        #   steering ∈ [-1, 1] → scaled to [-max_angular_vel, +max_angular_vel]
        #
        # PPO learns independently:
        #   throttle=0, steering=±1 → pure in-place rotation
        #   throttle=1, steering=0  → full speed straight
        #   throttle=0.5, steering=0.3 → slow curve
        throttle = torch.clamp(actions[:, 0], 0.0, 1.0)  # forward only
        steering = actions[:, 1]
        
        lin_vel = throttle * self.cfg.max_linear_vel       # [0, 0.3] m/s
        ang_vel = steering * self.cfg.max_angular_vel      # [-1.5, 1.5] rad/s

        # Differential drive IK
        left_vel  = (lin_vel - ang_vel * self.cfg.half_track) / self.cfg.wheel_radius
        right_vel = (lin_vel + ang_vel * self.cfg.half_track) / self.cfg.wheel_radius
        left_vel  = torch.clamp(left_vel,  -65.0, 65.0)
        right_vel = torch.clamp(right_vel, -65.0, 65.0)

        four_wheel_actions = torch.zeros((self.num_envs, 4), device=self.device)
        four_wheel_actions[:, 0] = left_vel
        four_wheel_actions[:, 1] = left_vel
        four_wheel_actions[:, 2] = right_vel
        four_wheel_actions[:, 3] = right_vel
        self.robot.set_joint_velocity_target(four_wheel_actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        self.root_pos = self.robot.data.root_pos_w[:, :2]
        self.forwards = math_utils.quat_apply(
            self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B
        )[:, :2]

        # Determine current navigation subgoal (next waypoint or final target)
        subgoal = self._get_current_subgoal()

        target_vector = subgoal - self.root_pos
        distance_to_goal = torch.norm(target_vector, dim=-1)

        # distance_to_target must be to the FINAL target, not the intermediate waypoint
        final_target_vec = self.target_pos[:, :2] - self.root_pos
        self.distance_to_target = torch.norm(final_target_vec, dim=-1)

        target_direction = target_vector / (distance_to_goal.unsqueeze(-1) + 1e-6)

        dot = torch.sum(self.forwards * target_direction, dim=-1, keepdim=True)
        cross = (self.forwards[:, 0] * target_direction[:, 1] -
                self.forwards[:, 1] * target_direction[:, 0]).unsqueeze(-1)
        distance_obs = distance_to_goal.unsqueeze(-1)

        # Add body-frame velocities so policy knows its own motion state
        fwd_vel = self.robot.data.root_com_lin_vel_b[:, 0:1]   # body X velocity
        ang_vel = self.robot.data.root_com_ang_vel_b[:, 2:3]   # yaw rate
        policy_obs = torch.hstack((dot, cross, distance_obs, fwd_vel, ang_vel))
        policy_obs = torch.nan_to_num(policy_obs, nan=0.0, posinf=100.0, neginf=-100.0)

        # ---- depth images → obstacle-line rays (pure vision) ----
        front_depth_raw = self._camera_front.data.output["distance_to_image_plane"]  # (B,12,200,1)
        back_depth_raw  = self._camera_back.data.output["distance_to_image_plane"]   # (B,12,200,1)
        front_depth_raw = torch.nan_to_num(front_depth_raw, nan=10.0, posinf=10.0, neginf=0.0)
        back_depth_raw  = torch.nan_to_num(back_depth_raw,  nan=10.0, posinf=10.0, neginf=0.0)
        # Replace 0.0 (no-hit / sky pixels) with 10.0 to prevent false near-clip
        front_depth_raw = torch.where(front_depth_raw <= 0.01, 10.0, front_depth_raw)
        back_depth_raw  = torch.where(back_depth_raw  <= 0.01, 10.0, back_depth_raw)
        # Keep raw metres copy for ray extraction (aligned with Rerun depth map)
        front_depth_m = torch.clamp(front_depth_raw, 0.0, 10.0)
        back_depth_m  = torch.clamp(back_depth_raw,  0.0, 10.0)
        # Normalised copy for model input
        front_depth_raw = front_depth_m / 10.0
        back_depth_raw  = back_depth_m / 10.0

        # ---- waypoint vector (next 2 waypoints in robot-local frame) ----
        waypoint_vec = self._get_waypoint_obs()  # (B, 4)
        waypoint_vec = torch.nan_to_num(waypoint_vec, nan=0.0, posinf=100.0, neginf=-100.0)

        # ---- full observation dict (multiple groups) ----
        # Extract obstacle-line rays from depth maps (stored for reuse in _get_rewards)
        self._lidar_rays = self._depth_to_rays(front_depth_m, back_depth_m)  # (B, 64)  raw metres

        # Update lidar ray markers in Isaac Sim viewport
        if self.render_mode != "headless" and self.cfg.is_ray_visualization:
            update_lidar_ray_markers(self)

        front_depth_2d = front_depth_raw.permute(0, 3, 1, 2)  # (B,1,12,200) for CNN models
        back_depth_2d  = back_depth_raw.permute(0, 3, 1, 2)

        return {
            "policy": policy_obs,            # (B, 5)  — 1D for MLP / RNN
            "waypoints": waypoint_vec,       # (B, 4)  — next 2 waypoints in robot frame
            "depth_front": front_depth_2d,   # (B, 1, 12, 200) — 2D for CNN
            "depth_back":  back_depth_2d,    # (B, 1, 12, 200)
            "lidar_rays":  self._lidar_rays,  # (B, 2, 64)      — 1D rays for MLP
        }

    def _get_rewards(self) -> torch.Tensor:
        """v35: GRALP-inspired distance_progress + aligned_forward + turn guide + obstacles."""
        # Init _rew_cnt early (used by obstacle_avoid_reward warmup)
        if not hasattr(self, '_rew_cnt'):
            self._rew_cnt = 0
        self._rew_cnt += 1

        thr = self.cfg.target_reach_threshold

        # ---- Subgoal direction & alignment ----
        subgoal_world = self._get_current_subgoal()
        subgoal_vec = subgoal_world - self.root_pos
        curr_subgoal_dist = torch.norm(subgoal_vec, dim=-1)
        target_dir = subgoal_vec / (curr_subgoal_dist.unsqueeze(-1) + 1e-6)
        alignment = torch.sum(self.forwards * target_dir, dim=-1)  # cos
        cross = (self.forwards[:, 0] * target_dir[:, 1] -
                 self.forwards[:, 1] * target_dir[:, 0])  # sin

        # ---- Velocities ----
        fwd_vel = self.robot.data.root_com_lin_vel_b[:, 0]
        ang_vel = self.robot.data.root_com_ang_vel_b[:, 2]

        # ---- Tilt ----
        quat = self.robot.data.root_link_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr, cosr)
        sinp = 2.0 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)

        # ---- Obstacle distance from lidar rays (stored by _get_observations) ----

        # ---- Distance progress (based on FINAL target — fixed reference) ----
        # Uses final target distance (self.distance_to_target) instead of volatile
        # subgoal distance, so the progress signal is stable even when subgoal
        # changes due to dynamic projection.
        if not hasattr(self, '_prev_final_dist'):
            self._prev_final_dist = self.distance_to_target.clone()
        prev_dist = self._prev_final_dist
        per_step_dt = self.step_dt * self.cfg.decimation
        progress = distance_progress_reward(
            prev_dist, self.distance_to_target, alignment,
            self.cfg.max_linear_vel, per_step_dt,
        )
        self._prev_final_dist = self.distance_to_target.clone()

        # ---- Stuck detection (shared by reward and dones) ----
        subgoal_dist_for_stuck = curr_subgoal_dist
        if not hasattr(self, "_stuck_counter"):
            self._stuck_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
            self._stuck_prev_dist = subgoal_dist_for_stuck.clone()
            self._stuck_episode_for_reward = self.episode_length_buf.clone()
        new_ep = self.episode_length_buf < self._stuck_episode_for_reward
        self._stuck_counter = torch.where(new_ep, torch.zeros_like(self._stuck_counter), self._stuck_counter)
        self._stuck_prev_dist = torch.where(new_ep, subgoal_dist_for_stuck, self._stuck_prev_dist)
        self._stuck_episode_for_reward = self.episode_length_buf.clone()
        stuck_step = (self._stuck_prev_dist - subgoal_dist_for_stuck).abs() < 0.01
        self._stuck_counter = torch.where(stuck_step, self._stuck_counter + 1, torch.zeros_like(self._stuck_counter))
        self._stuck_prev_dist = subgoal_dist_for_stuck.clone()
        stuck_mask = self._stuck_counter > 3600

        # ---- Assemble reward (v63: restored from v2) ----
        total = (
            alignment_reward(alignment)
            + idle_penalty(fwd_vel)
            + progress
            + speed_reward(fwd_vel, alignment)
            + misalign_speed_penalty(fwd_vel, alignment)
            + steer_toward_target_reward(cross, ang_vel, alignment)
            + yaw_damping(ang_vel)
            + obstacle_avoid_reward(self._lidar_rays)
            + waypoint_reach(curr_subgoal_dist)
            + tilt_penalty(roll, pitch)
            + backward_drive_penalty(fwd_vel)
            + stuck_penalty(stuck_mask)
        )
        total = torch.clamp(total, -20.0, 500.0)
        total = torch.nan_to_num(total, nan=0.0, posinf=10.0, neginf=-10.0)
        # v65: scale rewards to prevent value function gradient explosion
        total = total * self.cfg.reward_scale

        if self._rew_cnt % 200 == 1:
            dbg(f"R#{self._rew_cnt}: total={total.mean().item():.2f} prog={progress.mean().item():.2f} align={alignment.mean().item():.2f} fwd={fwd_vel.mean().item():.2f} dist={curr_subgoal_dist.mean().item():.2f} obs_min={self._lidar_rays.min().item():.2f} reach={((curr_subgoal_dist < thr).sum().item())}/{self.num_envs}")

        return total
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        reached = self.distance_to_target < self.cfg.target_reach_threshold

        # ---- Stuck detection (reuses _stuck_counter from _get_rewards) ----
        stuck = getattr(self, "_stuck_counter", torch.zeros(self.num_envs, device=self.device)) > 3600

        # ---- Collision (obstacle too close) ----
        env_origin_0 = self.scene.env_origins[0, :2]
        robot_xy_world = self.root_pos
        collision_list = []
        for env_id in range(self.num_envs):
            rx = robot_xy_world[env_id, 0].item() - env_origin_0[0].item()
            ry = robot_xy_world[env_id, 1].item() - env_origin_0[1].item()
            inside = self._is_inside_any_obstacle(rx, ry, 0, safe_margin=0.10)
            collision_list.append(inside)
        collision = torch.tensor(collision_list, device=self.device, dtype=torch.bool)

        # ---- Tilt / Exploding ----
        quat = self.robot.data.root_link_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        sinp = 2.0 * (w * y - z * x)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        tilted = (torch.abs(roll) > 0.5) | (torch.abs(pitch) > 0.5)
        exploding = torch.norm(self.robot.data.root_com_lin_vel_b, dim=-1) > 12.0

        # ---- No valid path found ----
        no_path = getattr(self, "_global_path", None) is None

        # ---- Per-robot death mask ----
        per_robot_dead = reached | stuck | collision | tilted | exploding | no_path

        # Track per-robot death (zombie state: dead robot stays frozen)
        if not hasattr(self, "_robot_dead"):
            self._robot_dead = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._robot_dead = self._robot_dead | per_robot_dead

        # ---- Global reset: only when ALL robots are dead OR episode timed out ----
        all_dead = self._robot_dead.all()
        global_reset = all_dead | time_out.any()

        # ---- Debug: log death reasons on global reset ----
        if global_reset:
            n_stuck = stuck.sum().item()
            n_collision = collision.sum().item()
            n_tilted = tilted.sum().item()
            n_reached = reached.sum().item()
            n_explode = exploding.sum().item()
            n_nopath = 32 if no_path else 0
            n_dead = self._robot_dead.sum().item()
            avg_ep_len = self.episode_length_buf.float().mean().item()
            max_ep = self.max_episode_length
            dist_to_target = self.distance_to_target.mean().item()
            dbg(f"DONE: trigger=timeout={time_out.any().item()} allDead={all_dead.item()} "
                f"nDead={n_dead}/32 maxEpSteps={max_ep} avgEpLen={avg_ep_len:.0f} "
                f"distToTgt={dist_to_target:.2f} "
                f"stuck={n_stuck} coll={n_collision} tilt={n_tilted} "
                f"reach={n_reached} explode={n_explode} nopath={n_nopath}")

        # terminated: True for ALL envs only on global reset
        terminated = torch.full((self.num_envs,), global_reset.item(), device=self.device, dtype=torch.bool)
        truncated = time_out

        # Clear dead mask on global reset
        if global_reset:
            self._robot_dead.zero_()

        return terminated, truncated

    def _reset_idx(self, env_ids: Optional[Sequence[int]]):
        if env_ids is None:
            _env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            _env_ids = env_ids
        else:
            _env_ids = torch.tensor(list(env_ids), dtype=torch.long, device=self.device)
        _env_ids = cast(torch.Tensor, _env_ids)

        if len(_env_ids) == 0:
            return

        half = self.cfg.arena_side_length / 2.0
        margin = self.cfg.wall_safe_margin

        super()._reset_idx(_env_ids)

        # ---- Fixed layout mode: only generate obstacles/start/target once ----
        _fixed_mode = not self.cfg.randomize_obstacles
        _is_first_reset = not getattr(self, '_fixed_layout_saved', False)

        if not _fixed_mode or _is_first_reset:
            # Regenerate shared obstacles
            if 0 in _env_ids:
                update_obstacles(self, torch.tensor([0], device=self.device), half, margin)
                scene_setup.hide_robots_from_depth_cameras()
            # All envs share the same obstacle data
            if 0 in self._obstacle_data:
                for env_id in range(self.num_envs):
                    self._obstacle_data[env_id] = self._obstacle_data[0]

            # Generate robot start position
            try:
                robot_positions = generate_random_robot_position(self, _env_ids[:1], half, margin)
                shared_rx = float(robot_positions[0, 0].item())
                shared_ry = float(robot_positions[0, 1].item())
            except Exception:
                shared_rx, shared_ry = 0.0, 0.0

            # Generate target position
            if 0 in _env_ids:
                robot_xy = self.robot.data.root_pos_w[_env_ids, :2]
                single_target = generate_random_target_position(self,
                    torch.tensor([0], device=self.device),
                    robot_xy[:1],
                    self.scene.env_origins[0, :2].unsqueeze(0),
                    half, margin)
                # Save for fixed-layout reuse
                self._fixed_target = single_target[0].clone()
                self._fixed_start = (shared_rx, shared_ry)

            # Plan global path
            self._global_path = None
            try:
                if 0 in self._occupancy_grids:
                    _grid, ox, oy, res = self._occupancy_grids[0]
                    goal_lx = single_target[0, 0].item() - self.scene.env_origins[0, 0].item()
                    goal_ly = single_target[0, 1].item() - self.scene.env_origins[0, 1].item()
                    path = plan_global_path(self, 0, ox, oy, res, (shared_rx, shared_ry), (goal_lx, goal_ly),
                                           use_astar=self.cfg.planner_use_astar)
                    if path is not None and len(path) >= 2:
                        self._global_path = path
                        if _fixed_mode:
                            self._fixed_path = self._global_path
            except Exception:
                pass

            if _fixed_mode:
                self._fixed_layout_saved = True
        else:
            # Fixed layout: reuse saved start, target, path
            shared_rx, shared_ry = self._fixed_start
            single_target = self._fixed_target.unsqueeze(0)
            self._global_path = getattr(self, '_fixed_path', None)

        default_root_state = self.robot.data.default_root_state[_env_ids].clone()
        init_z = self.cfg.robot_cfg.init_state.pos[2]
        # env_spacing=0 → all env_origins identical. Place every robot at SAME world coords.
        wx = self.scene.env_origins[0, 0].item() + shared_rx
        wy = self.scene.env_origins[0, 1].item() + shared_ry
        wz = self.scene.env_origins[0, 2].item() + init_z
        default_root_state[:, 0] = wx
        default_root_state[:, 1] = wy
        default_root_state[:, 2] = wz
        # Random yaw per robot (same start pos, different heading)
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
                    rwx = default_root_state[i, 0].item()
                    rwy = default_root_state[i, 1].item()
                    rwz = default_root_state[i, 2].item()
                    qw = default_root_state[i, 3].item()
                    qx = default_root_state[i, 4].item()
                    qy = default_root_state[i, 5].item()
                    qz = default_root_state[i, 6].item()
                    parent = prim.GetParent()
                    if parent.IsValid() and parent.GetPath() != Sdf.Path.absoluteRootPath:
                        xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                        world_mat = Gf.Matrix4d()
                        world_mat.SetTranslateOnly(Gf.Vec3d(rwx, rwy, rwz))
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
                        rwz_safe = max(rwz, float(init_z))
                        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(rwx, rwy, rwz_safe))
                        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(qw, qx, qy, qz))

        # Write robot root state
        self.robot.write_root_state_to_sim(default_root_state, _env_ids)

        # ---- Build target_positions from saved or newly-generated target ----
        target_positions = single_target.repeat(len(_env_ids), 1)
        self.target_pos[_env_ids, 0] = target_positions[:, 0]
        self.target_pos[_env_ids, 1] = target_positions[:, 1]
        self.target_pos[_env_ids, 2] = 0.0
        robot_xy = self.robot.data.root_pos_w[_env_ids, :2]
        target_vector = self.target_pos[_env_ids, :2] - robot_xy
        self.target_yaws[_env_ids] = torch.atan2(target_vector[:, 1], target_vector[:, 0]).unsqueeze(-1)

        # ---- Initialize per-robot replanning state ----
        if not hasattr(self, "_replan_cooldown") or self._replan_cooldown.shape[0] != self.num_envs:
            self._replan_cooldown = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        else:
            self._replan_cooldown[_env_ids] = 0

        # ---- Track final-target distance for progress reward & stuck detection ----
        fv = self.target_pos[_env_ids, :2] - robot_xy
        fv_dist = torch.norm(fv, dim=-1)
        if not hasattr(self, '_prev_final_dist') or self._prev_final_dist.shape[0] != self.num_envs:
            self._prev_final_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_final_dist[_env_ids] = fv_dist
        # Init waypoint distance tracker for continuous approach reward
        if not hasattr(self, '_prev_wp_dist') or self._prev_wp_dist.shape[0] != self.num_envs:
            self._prev_wp_dist = torch.zeros(self.num_envs, device=self.device)
        self._prev_wp_dist[_env_ids] = torch.norm(
            self._get_current_subgoal()[_env_ids] - self.robot.data.root_pos_w[_env_ids, :2], dim=-1)

        self._visualize_markers(robot_world_pos=default_root_state[:, :3])
        self._visualize_path_markers()

    # ===================== 导航 — 路径投影 & 子目标 =====================

    def _project_onto_path(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Project each robot onto the global path (cached per step)."""
        # Return cached result if already computed this step
        cached_idx = getattr(self, "_cached_closest_idx", None)
        cached_ct = getattr(self, "_cached_cross_track", None)
        if cached_idx is not None and cached_ct is not None:
            return cached_idx, cached_ct

        path = getattr(self, "_global_path", None)
        if not has_valid_path(path):
            zeros_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            zeros_ct = torch.zeros(self.num_envs, device=self.device)
            self._cached_closest_idx = zeros_idx
            self._cached_cross_track = zeros_ct
            return zeros_idx, zeros_ct

        waypoints_world = waypoints_local_to_world(
            path, self.scene.env_origins[0, :2], self.device,
        )
        closest_idx, cross_track = project_onto_path(
            self.robot.data.root_pos_w[:, :2], waypoints_world,
        )
        self._cached_closest_idx = closest_idx
        self._cached_cross_track = cross_track
        return closest_idx, cross_track

    def _get_current_subgoal(self) -> torch.Tensor:
        """Dynamic projection: subgoal = nearest waypoint + K look-ahead."""
        path = getattr(self, "_global_path", None)
        cached_idx = getattr(self, "_cached_closest_idx", None)
        subgoals, closest_idx = get_current_subgoal(
            self.robot.data.root_pos_w[:, :2],
            path,
            self.target_pos[:, :2],
            self.scene.env_origins[0, :2],
            self.cfg.waypoint_look_ahead_dist,
            self.device,
            cached_idx,
        )
        self._cached_closest_idx = closest_idx
        return subgoals

    def _get_waypoint_obs(self) -> torch.Tensor:
        """Next 2 waypoints in robot-local frame via dynamic projection."""
        path = getattr(self, "_global_path", None)
        cached_idx = getattr(self, "_cached_closest_idx", None)
        obs, closest_idx = get_waypoint_obs(
            self.root_pos, path,
            self.scene.env_origins[0, :2],
            self.device,
            cached_idx,
        )
        self._cached_closest_idx = closest_idx
        return obs

    def _try_replan(self) -> None:
        """Check cross-track error; replan for deviated robots (rate-limited)."""
        path = getattr(self, "_global_path", None)
        if not has_valid_path(path):
            return

        self._replan_cooldown = decrement_cooldowns(self._replan_cooldown)
        _, cross_track = self._project_onto_path()
        need_replan = check_replan_needed(
            cross_track, self._replan_cooldown, self.cfg.replan_threshold,
        )
        if not need_replan.any():
            return

        replan_ids = torch.where(need_replan)[0]
        env_origin = self.scene.env_origins[0, :2]

        for env_id in replan_ids.tolist():
            rx = self.robot.data.root_pos_w[env_id, 0].item() - env_origin[0].item()
            ry = self.robot.data.root_pos_w[env_id, 1].item() - env_origin[1].item()
            tx = self.target_pos[env_id, 0].item() - env_origin[0].item()
            ty = self.target_pos[env_id, 1].item() - env_origin[1].item()

            if 0 in self._occupancy_grids:
                _grid, ox, oy, res = self._occupancy_grids[0]
                try:
                    new_path = plan_global_path(
                        self, env_id, ox, oy, res, (rx, ry), (tx, ty),
                        save_map=False, use_astar=self.cfg.planner_use_astar,
                    )
                    if new_path is not None and len(new_path) >= 2:
                        # Set cooldown only on success (not wasted on failure)
                        self._replan_cooldown[env_id] = self.cfg.replan_cooldown_steps
                        self._global_path = new_path
                        # Invalidate stale projection cache after path change
                        self._cached_closest_idx = None
                        self._cached_cross_track = None
                        break
                except Exception:
                    pass

    # ===================== 深度图 -> Ray 提取 =====================

    def _depth_to_rays(self, front_depth, back_depth):
        """Extract obstacle-line rays from front+back depth maps."""
        return depth_to_rays(front_depth, back_depth, self.cfg.n_rays)

    # ===================== 可视化工具 =====================

    def _visualize_cameras(self):
        if self.cfg.is_camera_log:
            log_camera_views(self)
        else:
            pass

    def _visualize_markers(self, robot_world_pos=None):
        if self.render_mode == "headless":
            return
        update_markers(self, robot_world_pos)

    def _visualize_path_markers(self):
        if self.render_mode == "headless" or not self.cfg.is_path_visualization:
            return
        update_path_markers(self)

    # ===================== 环境更新 =====================

    def _update_obstacles(self, env_ids, half, margin):
        update_obstacles(self, env_ids, half, margin)

    def _build_occupancy_grid(self, env_id, half, wall_margin, obs_margin):
        return build_occupancy_grid(self, env_id, half, wall_margin, obs_margin)

    def _generate_random_robot_position(self, env_ids, half, margin):
        return generate_random_robot_position(self, env_ids, half, margin)

    def _generate_random_target_position(self, env_ids, robot_xy, env_origin_xy, half, margin):
        return generate_random_target_position(self, env_ids, robot_xy, env_origin_xy, half, margin)

    def _grid_sample_free(self, grid, origin_x, origin_y, resolution, arena_half):
        return grid_sample_free(grid, origin_x, origin_y, resolution, arena_half)

    def _is_free_in_grid(self, grid, origin_x, origin_y, resolution, wx, wy):
        return is_free_in_grid(grid, origin_x, origin_y, resolution, wx, wy)
    
    def _dist_to_nearest_obstacle(self, x, y, env_id):
        return dist_to_nearest_obstacle(self, x, y, env_id)

    def _is_inside_any_obstacle(self, x, y, env_id, safe_margin=0.15):
        return is_inside_any_obstacle(self, x, y, env_id, safe_margin)

    def _check_overlap(self, env_id, x, y, sx, sy):
        return check_obstacle_overlap(self, env_id, x, y, sx, sy)
