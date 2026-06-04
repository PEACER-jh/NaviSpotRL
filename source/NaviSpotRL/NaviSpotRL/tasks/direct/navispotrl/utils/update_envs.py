"""Environment update tools: obstacle regeneration, occupancy grid, robot/target placement."""

from __future__ import annotations

import math
import os
import random
import torch
import cv2
import numpy as np
import omni.usd
import omni.physx

import isaaclab.sim as sim_utils


# ===================== Obstacle management =====================

def update_obstacles(env, env_ids: torch.Tensor, half: float, margin: float) -> None:
    """Delete old obstacle prims, recreate with new random params, then reload physics."""
    stage = omni.usd.get_context().get_stage()
    env_origins = env.scene.env_origins

    obs_params = []  # list of (env_id, i, prim_path, ox, oy, sx, sy, sh)

    for env_id in env_ids.tolist():
        env._obstacle_data[env_id] = []
        for i in range(env.cfg.num_obstacles):
            prim_path = f"/World/Shared/Obstacle_{i:02d}"
            sx = random.uniform(env.cfg.box_min_size, env.cfg.box_max_size)
            sy = random.uniform(env.cfg.box_min_size, env.cfg.box_max_size)
            sh = random.uniform(env.cfg.box_min_height, env.cfg.box_max_height)
            ox = 0.0
            oy = 0.0

            placed = False
            for reduce in [1.0, 0.85, 0.7, 0.55, 0.4]:
                _sx = sx * reduce
                _sy = sy * reduce
                for _ in range(200):
                    ox = random.uniform(-half + margin, half - margin)
                    oy = random.uniform(-half + margin, half - margin)
                    if not check_obstacle_overlap(env, env_id, ox, oy, _sx, _sy):
                        sx, sy = _sx, _sy
                        placed = True
                        break
                if placed:
                    break

            if reduce < 1.0:
                sh *= reduce

            env._obstacle_data[env_id].append(("box", ox, oy, sx, sy, sh))
            obs_params.append((env_id, i, prim_path, ox, oy, sx, sy, sh))

    # Delete all old prims
    for _, _, prim_path, _, _, _, _, _ in obs_params:
        old_prim = stage.GetPrimAtPath(prim_path)
        if old_prim.IsValid():
            stage.RemovePrim(prim_path)

    # Recreate with new params
    for _, _, prim_path, ox, oy, sx, sy, sh in obs_params:
        obs_cfg = sim_utils.CuboidCfg(
            size=(sx, sy, sh),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        )
        obs_cfg.func(prim_path, obs_cfg, translation=(ox, oy, sh / 2))

    # Reload physics and reinitialize robot articulation
    omni.physx.acquire_physx_interface().force_load_physics_from_usd()
    env.robot._initialize_impl()


# ===================== Occupancy Grid =====================

def build_occupancy_grid(
    env, env_id: int, half: float, wall_margin: float, obs_margin: float
) -> tuple[np.ndarray, float, float, float]:
    """Build 2D occupancy grid covering arena [-half, half]^2 with padding.

    - grid_raw:  thin wall frame (1 cell) + raw obstacles (all black in image).
    - grid_inflated: wall frame + wall_margin band + obstacles + obs_margin band.
      The wall_margin band appears as light-gray in ``occupancy_inflated.png``.

    Returns:
        grid: 2D bool array (True = occupied, after safety inflation).
        origin_x, origin_y: env-local coords of grid[0, 0].
        resolution: cell size in meters.
    """
    resolution = env.cfg.grid_resolution
    padding = max(0.5, wall_margin + obs_margin + 0.2)
    arena_span = 2.0 * half + 2.0 * padding
    cells = max(1, int(arena_span / resolution + 0.5))
    resolution = arena_span / cells
    origin_x = -half - padding
    origin_y = -half - padding

    grid_raw = np.zeros((cells, cells), dtype=bool)
    grid_inflated = np.zeros((cells, cells), dtype=bool)

    def world_to_grid(wx: float, wy: float) -> tuple[int, int]:
        gx = int((wx - origin_x) / resolution)
        gy = int((wy - origin_y) / resolution)
        return max(0, min(cells - 1, gx)), max(0, min(cells - 1, gy))

    # ---- thin wall frame (black border, 1 cell) ----
    arena_min = -half
    arena_max = half
    ax0, ay0 = world_to_grid(arena_min, arena_min)
    ax1, ay1 = world_to_grid(arena_max, arena_max)
    _draw_frame(grid_raw, ax0, ay0, ax1, ay1)
    _draw_frame(grid_inflated, ax0, ay0, ax1, ay1)

    # ---- wall inflation band (inside arena, light-gray in inflated) ----
    wm_in = world_to_grid(0, arena_min + wall_margin)[1]
    wm_out = world_to_grid(0, arena_max - wall_margin)[1]
    wm_left = world_to_grid(arena_min + wall_margin, 0)[0]
    wm_right = world_to_grid(arena_max - wall_margin, 0)[0]
    # bottom / top / left / right bands → only in grid_inflated
    grid_inflated[ay0 + 1:wm_in + 1, ax0:ax1 + 1] = True
    grid_inflated[wm_out:ay1, ax0:ax1 + 1] = True
    grid_inflated[wm_in + 1:wm_out, ax0:wm_left + 1] = True
    grid_inflated[wm_in + 1:wm_out, wm_right:ax1 + 1] = True

    # ---- obstacles ----
    if env_id in env._obstacle_data:
        for _, ox, oy, sx, sy, _ in env._obstacle_data[env_id]:
            # raw obstacles → grid_raw (black)
            _fill_grid_rect(grid_raw, ox, oy, sx / 2.0, sy / 2.0, cells, world_to_grid)
            # raw obstacles → grid_inflated (black core)
            _fill_grid_rect(grid_inflated, ox, oy, sx / 2.0, sy / 2.0, cells, world_to_grid)
            # inflated obstacles → grid_inflated (gray margin)
            _fill_grid_rect(grid_inflated, ox, oy,
                            sx / 2.0 + obs_margin, sy / 2.0 + obs_margin,
                            cells, world_to_grid)

    save_occupancy_grids(env, grid_raw, grid_inflated, origin_x, origin_y, resolution, cells)
    return grid_inflated, origin_x, origin_y, resolution


def _draw_frame(grid: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    """Draw a 1-cell rectangle frame on the grid."""
    grid[y0, x0:x1 + 1] = True
    grid[y1, x0:x1 + 1] = True
    grid[y0:y1 + 1, x0] = True
    grid[y0:y1 + 1, x1] = True


def _fill_grid_rect(grid: np.ndarray, cx: float, cy: float,
                    half_sx: float, half_sy: float, cells: int,
                    world_to_grid) -> None:
    """Fill a rectangular region in the grid."""
    igx_min, igy_min = world_to_grid(cx - half_sx, cy - half_sy)
    igx_max, igy_max = world_to_grid(cx + half_sx, cy + half_sy)
    igx_min = max(0, igx_min)
    igy_min = max(0, igy_min)
    igx_max = min(cells - 1, igx_max)
    igy_max = min(cells - 1, igy_max)
    if igx_min <= igx_max and igy_min <= igy_max:
        grid[igy_min:igy_max + 1, igx_min:igx_max + 1] = True


def save_occupancy_grids(env, grid_raw: np.ndarray, grid_inflated: np.ndarray,
                         origin_x: float, origin_y: float, resolution: float,
                         cells: int) -> None:
    """Save raw and inflated occupancy grids as pair of PNG images to utils/map/."""
    if os.environ.get("NAVISPOTRL_SAVE_MAP", "1") != "1":
        return
    map_dir = os.path.join(os.path.dirname(__file__), "map")
    os.makedirs(map_dir, exist_ok=True)

    h, w = grid_raw.shape
    img_raw = np.where(grid_raw, 0, 255).astype(np.uint8)[::-1, :]
    cv2.imwrite(os.path.join(map_dir, "occupancy_raw.png"), img_raw)

    margin_only = grid_inflated & (~grid_raw)
    img_infl = np.full((h, w), 255, dtype=np.uint8)
    img_infl[grid_inflated] = 0
    img_infl[margin_only] = 128
    img_infl = img_infl[::-1, :]
    cv2.imwrite(os.path.join(map_dir, "occupancy_inflated.png"), img_infl)

def grid_sample_free(grid: np.ndarray, origin_x: float, origin_y: float,
                      resolution: float, arena_half: float) -> tuple[float | None, float | None]:
    """Sample a random free cell within arena [-arena_half, arena_half], return env-local (x, y)."""
    free_ys, free_xs = np.where(~grid)
    if len(free_ys) == 0:
        return None, None
    # Shuffle to avoid bias toward top-left when filtering
    order = list(range(len(free_ys)))
    random.shuffle(order)
    for idx in order:
        cx = origin_x + (free_xs[idx] + 0.5) * resolution
        cy = origin_y + (free_ys[idx] + 0.5) * resolution
        if -arena_half <= cx <= arena_half and -arena_half <= cy <= arena_half:
            return cx, cy
    return None, None


def is_free_in_grid(grid: np.ndarray, origin_x: float, origin_y: float,
                    resolution: float, wx: float, wy: float) -> bool:
    """Check if a world-local point falls on a free cell."""
    gx = int((wx - origin_x) / resolution)
    gy = int((wy - origin_y) / resolution)
    if gx < 0 or gx >= grid.shape[1] or gy < 0 or gy >= grid.shape[0]:
        return False
    return not grid[gy, gx]


# ===================== Robot & Target Placement =====================

def generate_random_robot_position(
    env, env_ids: torch.Tensor, half: float, margin: float
) -> torch.Tensor:
    """Generate robot positions from free cells in occupancy grid."""
    positions = torch.zeros((len(env_ids), 2), device=env.device)
    wall_margin = margin
    obs_margin = env.cfg.obstacle_safe_margin

    for i, env_id in enumerate(env_ids.tolist()):
        grid, ox, oy, res = build_occupancy_grid(env, env_id, half, wall_margin, obs_margin)
        env._occupancy_grids[env_id] = (grid, ox, oy, res)

        sample = grid_sample_free(grid, ox, oy, res, half)
        if sample[0] is not None and sample[1] is not None:
            positions[i, 0] = sample[0]  # type: ignore[assignment]
            positions[i, 1] = sample[1]  # type: ignore[assignment]
        else:
            print(f"[Warning] env={env_id} no free cell for robot, using (0,0)")
            positions[i, 0] = 0.0
            positions[i, 1] = 0.0

    return positions


def generate_random_target_position(
    env, env_ids: torch.Tensor, robot_xy: torch.Tensor,
    env_origin_xy: torch.Tensor, half: float, margin: float,
) -> torch.Tensor:
    """Generate target positions from free cells within target_distance_range.

    Strategy:
    1. Sample free cells from grid, keep those satisfying distance range (2000 tries).
    2. Fallback: random (dist, angle) on the annulus, check grid (1000 tries).
    3. Final fallback: scan concentric rings relaxing distance by 20%.
    """
    target_positions = torch.zeros((len(env_ids), 2), device=env.device)
    wall_margin = margin
    obs_margin = env.cfg.obstacle_safe_margin
    dist_min, dist_max = env.cfg.target_distance_range

    for i, env_id in enumerate(env_ids.tolist()):
        grid, ox, oy, res = build_occupancy_grid(env, env_id, half, wall_margin, obs_margin)
        robot_lx: float = float(robot_xy[i, 0].item()) - float(env_origin_xy[i, 0].item())
        robot_ly: float = float(robot_xy[i, 1].item()) - float(env_origin_xy[i, 1].item())

        # ---- Stage 1: grid free-sample + distance filter ----
        found = False
        for _ in range(2000):
            sample = grid_sample_free(grid, ox, oy, res, half)
            if sample[0] is None:
                break
            cx: float = sample[0]  # type: ignore[assignment]
            cy: float = sample[1]  # type: ignore[assignment]
            dx = cx - robot_lx
            dy = cy - robot_ly
            dist = math.hypot(dx, dy)
            if dist_min <= dist <= dist_max:
                target_positions[i, 0] = float(env_origin_xy[i, 0].item()) + cx
                target_positions[i, 1] = float(env_origin_xy[i, 1].item()) + cy
                found = True
                break

        # ---- Stage 2: random annulus + grid check ----
        if not found:
            for _ in range(1000):
                dist = random.uniform(dist_min, dist_max)
                angle = random.uniform(-math.pi, math.pi)
                tx = robot_lx + dist * math.cos(angle)
                ty = robot_ly + dist * math.sin(angle)
                if -half <= tx <= half and -half <= ty <= half and is_free_in_grid(grid, ox, oy, res, tx, ty):
                    target_positions[i, 0] = float(env_origin_xy[i, 0].item()) + tx
                    target_positions[i, 1] = float(env_origin_xy[i, 1].item()) + ty
                    found = True
                    break

        # ---- Stage 3: ring scan with relaxed distance ----
        if not found:
            relaxed_min = dist_min * 0.7
            relaxed_max = dist_max * 1.3
            n_steps = 60
            for outer in range(n_steps):
                r = relaxed_min + outer * (relaxed_max - relaxed_min) / (n_steps - 1)
                n_angles = max(12, int(2 * math.pi * r / res))
                for ai in range(n_angles):
                    angle = 2 * math.pi * ai / n_angles
                    tx = robot_lx + r * math.cos(angle)
                    ty = robot_ly + r * math.sin(angle)
                    if -half <= tx <= half and -half <= ty <= half and is_free_in_grid(grid, ox, oy, res, tx, ty):
                        dist = math.hypot(tx - robot_lx, ty - robot_ly)
                        if dist_min * 0.6 <= dist <= dist_max * 1.5:
                            target_positions[i, 0] = float(env_origin_xy[i, 0].item()) + tx
                            target_positions[i, 1] = float(env_origin_xy[i, 1].item()) + ty
                            found = True
                            break
                if found:
                    break

        if not found:
            print(f"[Error] env={env_id} absolutely no safe target position!")
            target_positions[i, 0] = float(robot_xy[i, 0].item())
            target_positions[i, 1] = float(robot_xy[i, 1].item())

    return target_positions

def dist_to_nearest_obstacle(env, x: float, y: float, env_id: int) -> float:
    """Return distance from (x,y) to nearest obstacle edge (>=0 means outside)."""
    if env_id not in env._obstacle_data:
        return 1e9
    min_dist = 1e9
    for _, ox, oy, sx, sy, _ in env._obstacle_data[env_id]:
        dx = abs(x - ox) - sx / 2.0
        dy = abs(y - oy) - sy / 2.0
        if dx < 0 and dy < 0:
            return float(max(dx, dy))  # inside
        outside = max(dx, 0.0) + max(dy, 0.0)
        if outside < min_dist:
            min_dist = outside
    return float(min_dist)


def is_inside_any_obstacle(env, x: float, y: float, env_id: int, safe_margin: float = 0.15) -> bool:
    """Check if (x,y) (env-local) lies inside any inflated obstacle."""
    if env_id not in env._obstacle_data:
        return False
    for _, ox, oy, sx, sy, _ in env._obstacle_data[env_id]:
        half_sx = sx / 2.0 + safe_margin
        half_sy = sy / 2.0 + safe_margin
        if abs(x - ox) < half_sx and abs(y - oy) < half_sy:
            return True
    return False


def check_obstacle_overlap(env, env_id: int, x: float, y: float, sx: float, sy: float) -> bool:
    """Check if a new obstacle at (x,y) with size (sx,sy) overlaps existing ones."""
    if env_id not in env._obstacle_data:
        return False
    for _, ox, oy, osx, osy, _ in env._obstacle_data[env_id]:
        if abs(x - ox) < (sx + osx) * 0.5 and abs(y - oy) < (sy + osy) * 0.5:
            return True
    return False
