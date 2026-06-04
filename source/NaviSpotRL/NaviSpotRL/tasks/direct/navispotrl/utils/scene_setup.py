"""Scene setup utilities: walls, obstacles, depth cameras, markers.

Extracted from NavispotrlEnv._setup_scene to keep the env file clean.
"""

from __future__ import annotations

import random
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg

from .visualize_tools import define_markers


# ===================== 竞技场墙壁 =====================

def spawn_arena_walls(half: float, wall_height: float, wall_thickness: float) -> None:
    """Create four kinematic walls forming a square arena.

    Walls are placed under /World/Shared/ so all parallel envs share them.

    Args:
        half: Half of arena side length [m].
        wall_height: Z-height of each wall [m].
        wall_thickness: Wall thickness along the normal direction [m].
    """
    wall_len = 2.0 * half + wall_thickness
    wall_cfgs = [
        ("/World/Shared/Wall_Right", (half, 0.0, wall_height / 2), (wall_thickness, wall_len, wall_height)),
        ("/World/Shared/Wall_Left", (-half, 0.0, wall_height / 2), (wall_thickness, wall_len, wall_height)),
        ("/World/Shared/Wall_Front", (0.0, half, wall_height / 2), (wall_len, wall_thickness, wall_height)),
        ("/World/Shared/Wall_Back", (0.0, -half, wall_height / 2), (wall_len, wall_thickness, wall_height)),
    ]
    for prim_path, pos, size in wall_cfgs:
        cfg = sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        )
        cfg.func(prim_path, cfg, translation=pos)


# ===================== 障碍物 =====================

def spawn_shared_obstacles(
    num_obstacles: int,
    half: float,
    wall_safe_margin: float,
    box_min_size: float,
    box_max_size: float,
    box_min_height: float,
    box_max_height: float,
) -> None:
    """Spawn random non-overlapping box obstacles under /World/Shared/.

    Obstacles are dynamic rigid bodies with very high mass (effectively
    immovable). Positions and sizes are chosen randomly with collision
    avoidance between obstacles.

    Args:
        num_obstacles: Number of obstacles to spawn.
        half: Half of arena side length [m].
        wall_safe_margin: Minimum distance from arena walls [m].
        box_min_size, box_max_size: Side length range [m].
        box_min_height, box_max_height: Height range [m].
    """
    temp_obs: list[tuple[float, float, float, float]] = []
    for i in range(num_obstacles):
        sx = random.uniform(box_min_size, box_max_size)
        sy = random.uniform(box_min_size, box_max_size)
        sh = random.uniform(box_min_height, box_max_height)

        for _ in range(200):
            ox = random.uniform(-half + wall_safe_margin, half - wall_safe_margin)
            oy = random.uniform(-half + wall_safe_margin, half - wall_safe_margin)
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
            f"/World/Shared/Obstacle_{i:02d}",
            obs_cfg,
            translation=(ox, oy, sh / 2),
        )


# ===================== 深度相机 =====================

def create_front_depth_camera() -> Camera:
    """Create a front-facing wide-FOV depth camera.

    Specifications:
      - Resolution: 200×12 (≈16.7:1 — matches sensor for 136°×17° FOV)
      - HFOV: ~136° (horizontal_aperture=25.0, focal_length=5.0)
      - Range: 0.05–10 m
    """
    camera_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panoramic_rover/base_link/camera_front_link/CameraFront",
        update_period=0.032,
        height=12,
        width=200,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5.0,
            horizontal_aperture=25.0,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.00, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0),
            convention="world",
        ),
    )
    return Camera(cfg=camera_cfg)


def create_back_depth_camera() -> Camera:
    """Create a rear-facing wide-FOV depth camera.

    Same optics as the front camera, mounted on camera_back_link.
    """
    camera_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panoramic_rover/base_link/camera_back_link/CameraBack",
        update_period=0.032,
        height=12,
        width=200,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5.0,
            horizontal_aperture=25.0,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.00, 0.0, 0.00),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )
    return Camera(cfg=camera_cfg)


# ===================== 机器人深度可见性 =====================

def hide_robots_from_depth_cameras() -> None:
    """Mark all Robot mesh prims as invisible to secondary rays (depth cameras).

    USD's ``primvars:invisibleToSecondaryRays`` prevents meshes from being
    sampled by depth/reflection rays, while leaving them visible to primary
    (viewport) rays.  This ensures robots don't see each other in depth images.
    """
    import omni.usd
    from pxr import UsdGeom, Sdf
    import isaaclab.sim as _sim_utils

    stage = omni.usd.get_context().get_stage()
    # Walk all prims under the robot namespace
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        if "/Robot/" in path and prim.IsA(UsdGeom.Gprim):
            _sim_utils.change_prim_property(
                prop_path=f"{path}.primvars:invisibleToSecondaryRays",
                value=True,
                stage=stage,
                type_to_create_if_not_exist=Sdf.ValueTypeNames.Bool,
            )


# ===================== 标记 & 照明 =====================

def setup_markers_and_lights(env) -> None:
    """Set up dome light and visualization markers on the env instance.

    Also initialises per-env marker offset tensors so that markers
    are visible from the start (before the first reset).
    """
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    env.visualization_markers = define_markers()
    env.up_dir = torch.tensor([0.0, 0.0, 1.0], device=env.device)

    # Default target position so markers never disappear
    env.target_pos = env.scene.env_origins[:, :3] + torch.tensor(
        [0.0, 0.0, 0.2], device=env.device
    )
    env.target_yaws = torch.zeros((env.num_envs, 1), device=env.device)
    env._marker_offset = torch.zeros((env.num_envs, 3), device=env.device)
    env._marker_offset[:, -1] = 0.5
    env._target_point_offset = torch.zeros((env.num_envs, 3), device=env.device)
    env._target_point_offset[:, -1] = 0.2
