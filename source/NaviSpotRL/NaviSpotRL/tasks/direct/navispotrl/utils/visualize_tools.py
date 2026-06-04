"""Visualization tools for NaviSpotRL: markers, camera streaming, and robot USD sync."""

from __future__ import annotations

import cv2
import torch
import numpy as np
import rerun as rr
from pxr import Usd, UsdGeom, Gf, Sdf
import omni.usd

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


# ===================== Markers (arrows + target sphere) =====================

def define_markers() -> VisualizationMarkers:
    """Define visualization markers: cyan=robot forward, orange=target direction, red sphere=target point."""
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


def update_markers(
    env,
    robot_world_pos: torch.Tensor | None = None,
) -> None:
    """Update marker positions to match current robot pose and target position.

    Args:
        env: The NaviSpotRL environment instance (for access to robot data, target, markers, etc.).
        robot_world_pos: Optional override for robot root position (used during _reset_idx).
    """
    robot_pos = robot_world_pos if robot_world_pos is not None else env.robot.data.root_pos_w
    forward_quat = env.robot.data.root_quat_w
    target_vector = env.target_pos[:, :2] - robot_pos[:, :2]
    target_yaws_real = torch.atan2(target_vector[:, 1], target_vector[:, 0]).unsqueeze(-1)
    target_quat = math_utils.quat_from_angle_axis(target_yaws_real, env.up_dir).squeeze()

    forward_loc = robot_pos + env._marker_offset
    target_loc = robot_pos + env._marker_offset
    target_point_loc = env.target_pos + env._target_point_offset
    target_point_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)

    if target_quat.dim() == 1:
        target_quat = target_quat.unsqueeze(0)

    loc = torch.vstack((forward_loc, target_loc, target_point_loc))
    rots = torch.vstack((forward_quat, target_quat, target_point_rot))

    all_envs = torch.arange(env.cfg.scene.num_envs, device=env.device)
    indices = torch.hstack((
        torch.zeros_like(all_envs),
        torch.ones_like(all_envs),
        2 * torch.ones_like(all_envs),
    ))
    indices = indices.to(dtype=torch.int32)

    env.visualization_markers.visualize(loc, rots, marker_indices=indices)


# ===================== Robot USD Pose Sync (for non-fabric rendering) =====================



# ===================== Path Visualization (topological plan) =====================

def update_path_markers(env) -> None:
    """Draw global path waypoints as small spheres and line segments in the USD stage.

    Reads ``env._global_path`` (list of (x,y) env-local waypoints) and places:
      - Small green spheres at intermediate waypoints (skip start=robot, goal=target).
      - Yellow lines connecting consecutive waypoints from robot → goal.
    """
    path = getattr(env, "_global_path", None)
    if not path or len(path) < 2:
        _clear_path_prims(env)
        return

    stage = omni.usd.get_context().get_stage()
    env_origin = env.scene.env_origins[0, :3].cpu().tolist()
    robot_pos = env.robot.data.root_pos_w[0].cpu().tolist()
    target_pos = env.target_pos[0].cpu().tolist()

    # --- waypoint spheres (green, r=0.08) ---
    sphere_prims = []
    for i, (lx, ly) in enumerate(path):
        wx = env_origin[0] + lx
        wy = env_origin[1] + ly
        prim_path = f"/World/envs/env_0/PathSphere_{i:02d}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            prim = stage.DefinePrim(prim_path, "Sphere")
            prim.CreateAttribute("radius", Sdf.ValueTypeNames.Double).Set(0.08)
            prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
            prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(1, 1, 1))
            UsdGeom.Xformable(prim).AddTranslateOp()
            # Invisible to depth cameras
            prim.CreateAttribute("primvars:invisibleToSecondaryRays", Sdf.ValueTypeNames.Bool).Set(True)
            # Color: green via displayColor on a child prim
            mesh_prim = stage.DefinePrim(prim_path + "/mesh", "Sphere")
            mesh_prim.CreateAttribute("radius", Sdf.ValueTypeNames.Double).Set(0.08)
            mesh_prim.CreateAttribute("primvars:invisibleToSecondaryRays", Sdf.ValueTypeNames.Bool).Set(True)
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(wx, wy, 0.15))
        sphere_prims.append(prim_path)

    # Clean up any leftover spheres from a previous longer path
    i = len(path)
    while True:
        old_prim = stage.GetPrimAtPath(f"/World/envs/env_0/PathSphere_{i:02d}")
        if old_prim.IsValid():
            stage.RemovePrim(f"/World/envs/env_0/PathSphere_{i:02d}")
            i += 1
        else:
            break

    # --- path line segments (yellow) ---
    # Build segments: robot → waypoint_0 → waypoint_1 → ... → target
    vertices_list = []
    # robot position (world)
    vertices_list.append(Gf.Vec3f(float(robot_pos[0]), float(robot_pos[1]), 0.15))
    for lx, ly in path:
        wx = env_origin[0] + lx
        wy = env_origin[1] + ly
        vertices_list.append(Gf.Vec3f(float(wx), float(wy), 0.15))
    # target position
    vertices_list.append(Gf.Vec3f(float(target_pos[0]), float(target_pos[1]), 0.15))

    line_prim_path = "/World/envs/env_0/PathLine"
    line_prim = stage.GetPrimAtPath(line_prim_path)
    if not line_prim.IsValid():
        line_prim = stage.DefinePrim(line_prim_path, "BasisCurves")
        line_prim.CreateAttribute("type", Sdf.ValueTypeNames.Token).Set("linear")
        line_prim.CreateAttribute("curveVertexCounts", Sdf.ValueTypeNames.IntArray).Set([len(vertices_list)])
        line_prim.CreateAttribute("widths", Sdf.ValueTypeNames.FloatArray).Set([0.02])
        line_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
        line_prim.CreateAttribute("primvars:invisibleToSecondaryRays", Sdf.ValueTypeNames.Bool).Set(True)
        UsdGeom.Xformable(line_prim).AddTranslateOp()

    line_prim.GetAttribute("curveVertexCounts").Set([len(vertices_list)])
    line_prim.GetAttribute("points").Set(vertices_list)

    # Try to set line color (may not be supported in all viewers)
    color_attr = line_prim.GetAttribute("primvars:displayColor")
    if not color_attr.IsValid():
        color_attr = line_prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray)
    color_attr.Set([Gf.Vec3f(1.0, 0.8, 0.0)])  # yellow-orange


def _clear_path_prims(env) -> None:
    """Remove all path-related prims when no path is available."""
    stage = omni.usd.get_context().get_stage()
    i = 0
    while True:
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/PathSphere_{i:02d}")
        if prim.IsValid():
            stage.RemovePrim(f"/World/envs/env_0/PathSphere_{i:02d}")
            i += 1
        else:
            break
    line_prim = stage.GetPrimAtPath("/World/envs/env_0/PathLine")
    if line_prim.IsValid():
        stage.RemovePrim("/World/envs/env_0/PathLine")


def sync_robot_pose_to_usd(env) -> None:
    """Sync robot USD prim transforms with current PhysX state (needed when use_fabric=False)."""
    stage = omni.usd.get_context().get_stage()
    body_pose_w = env.robot.data.body_link_pose_w.detach().cpu().numpy()
    num_bodies = len(env.robot.body_names)

    with Sdf.ChangeBlock():
        for env_id in range(env.num_envs):
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
                body_name = env.robot.body_names[body_idx]
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
                    lz = 0.04

                try:
                    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(lx, ly, lz))
                    prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(lqw, lqx, lqy, lqz))
                except Exception:
                    pass


# ===================== Camera Visualization (Rerun streaming) =====================

def log_camera_views(env) -> None:
    """Log front/back camera RGB + depth to Rerun for live monitoring.

    Handles all combinations gracefully:
      1. RGB + depth both present → vertical stack (RGB top, depth bottom)
      2. Only RGB or only depth → single image log
      3. Only front camera → skip back
      4. Only back camera → skip front
      5. No cameras at all → no-op
    """
    def _get_image(camera, key):
        """Safely get a camera output as numpy array (first env only)."""
        try:
            data = camera.data.output.get(key)
            if data is None:
                return None
            arr = data[0].cpu().numpy()
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr.squeeze(-1)  # (H,W,1) -> (H,W)
            return arr
        except (AttributeError, KeyError, IndexError):
            return None

    def _build_view(rgb_np, depth_np):
        """Build a single view image from optional RGB and depth.

        Depth is displayed as grayscale (near=dark, far=white), not color-mapped,
        to reduce rendering overhead. When both RGB and depth are present,
        depth is resized to match RGB width before vertical stacking.
        """
        parts = []
        if rgb_np is not None:
            # Ensure RGB is uint8 0-255, shape (H, W) or (H, W, 3)
            if rgb_np.dtype != np.uint8:
                rgb_np = (np.clip(rgb_np, 0, 1) * 255).astype(np.uint8)
            if rgb_np.ndim == 2:
                rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_GRAY2BGR)

        if depth_np is not None:
            # Grayscale: clip 0-3m, near=white, far=black
            depth_clipped = np.clip(depth_np, 0.0, 3.0)
            depth_gray = ((1.0 - depth_clipped / 3.0) * 255).astype(np.uint8)
            depth_img = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)

        if rgb_np is not None and depth_np is not None:
            # Match widths for clean vertical stack
            if rgb_np.shape[1] != depth_img.shape[1]:
                depth_img = cv2.resize(depth_img, (rgb_np.shape[1], depth_img.shape[0]))
            return np.vstack([rgb_np, depth_img])
        elif rgb_np is not None:
            return rgb_np
        elif depth_np is not None:
            return depth_img
        else:
            return None

    # --- Front camera ---
    front_rgb = None
    front_depth = None
    if getattr(env, '_camera_front', None) is not None:
        front_rgb = _get_image(env._camera_front, "rgb")
        front_depth = _get_image(env._camera_front, "distance_to_image_plane")

    # --- Back camera ---
    back_rgb = None
    back_depth = None
    if getattr(env, '_camera_back', None) is not None:
        back_rgb = _get_image(env._camera_back, "rgb")
        back_depth = _get_image(env._camera_back, "distance_to_image_plane")

    # --- Build and log ---
    rr.set_time_sequence("step", getattr(env, '_rerun_step', 0))

    front_img = _build_view(front_rgb, front_depth)
    if front_img is not None:
        rr.log("camera/front", rr.Image(front_img))

    back_img = _build_view(back_rgb, back_depth)
    if back_img is not None:
        rr.log("camera/back", rr.Image(back_img))


# ===================== Lidar Rays Visualization (Isaac Sim USD) =====================

def init_lidar_ray_markers(env) -> None:
    """Create a VisualizationMarkers instance for lidar ray arrows."""
    import math
    import isaaclab.sim as _sim
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    env._ray_angles = []
    hfov = math.radians(136.0)
    n = 32
    for i in range(n):
        env._ray_angles.append(-hfov / 2 + i * hfov / (n - 1))
    for i in range(n):
        env._ray_angles.append(math.pi - hfov / 2 + i * hfov / (n - 1))

    env._ray_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/rayMarkers",
        markers={
            "ray": _sim.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(1.0, 1.0, 1.0),
                visual_material=_sim.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
            ),
        },
    )
    env._ray_markers = VisualizationMarkers(cfg=env._ray_marker_cfg)


def update_lidar_ray_markers(env) -> None:
    """Update ray arrows via VisualizationMarkers (PointInstancer)."""
    import math
    import torch
    rays = getattr(env, "_lidar_rays", None)
    if rays is None:
        return
    dists = rays[0].cpu().numpy()
    angs = getattr(env, "_ray_angles", None)
    if angs is None:
        return
    px = env.robot.data.root_pos_w[0, 0].item()
    py = env.robot.data.root_pos_w[0, 1].item()
    q = env.robot.data.root_quat_w[0].cpu().tolist()
    siny = 2.0 * (q[0] * q[3] + q[1] * q[2])
    cosy = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
    yaw = math.atan2(siny, cosy)

    locs = torch.zeros(64, 3, device=env.device)
    rots = torch.zeros(64, 4, device=env.device)
    scs  = torch.zeros(64, 3, device=env.device)
    for i in range(64):
        d = float(dists[i])
        d = min(max(d, 0.1), 3.0)
        a = angs[i] + yaw
        locs[i] = torch.tensor([px, py, 0.10])
        rots[i] = torch.tensor([math.cos(a/2), 0.0, 0.0, math.sin(a/2)])
        scs[i]  = torch.tensor([d, 0.06, 0.06])  # X=length, YZ=thickness
    indices = torch.zeros(64, dtype=torch.int32, device=env.device)
    env._ray_markers.visualize(locs, rots, scs, marker_indices=indices)