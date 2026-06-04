"""Navigation utilities: waypoint projection, subgoal selection, replanning.

All functions are pure tensor operations — they take state as explicit
arguments rather than accessing ``self``.  The env methods in
``NavispotrlEnv`` are thin wrappers that extract state and call these.

Extracted from ``NavispotrlEnv`` internal methods.
"""

from __future__ import annotations

import torch


# ===================== Path helpers =====================

def waypoints_local_to_world(
    path: list[tuple[float, float]] | None,
    env_origin: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Convert env-local waypoints to world coordinates.

    Args:
        path: List of (x, y) tuples in env-local frame.
        env_origin: (2,) tensor [ox, oy] — world position of env origin.
        device: Target torch device.

    Returns:
        (N, 2) tensor of waypoint world XY positions.
    """
    if path is None:
        raise ValueError("path must not be None")
    return torch.tensor(
        [[env_origin[0].item() + p[0], env_origin[1].item() + p[1]] for p in path],
        device=device,
    )


def has_valid_path(path: list | None) -> bool:
    """Return True if *path* is a list with at least 2 waypoints."""
    return path is not None and isinstance(path, list) and len(path) >= 2


# ===================== Path projection =====================

def project_onto_path(
    robot_pos: torch.Tensor,
    waypoints_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project each robot's position onto a path of waypoints.

    For each robot, finds the closest waypoint and computes the
    perpendicular (cross-track) distance to the nearest path segment.

    Args:
        robot_pos: (B, 2) world XY of each robot.
        waypoints_world: (N, 2) world XY of path waypoints.

    Returns:
        closest_idx: (B,) int32 — index of nearest waypoint on path.
        cross_track: (B,) float — perpendicular distance [m] from
            robot to the nearest path segment.
    """
    # Distance from each robot to each waypoint: (B, N)
    diff = robot_pos[:, None, :] - waypoints_world[None, :, :]  # (B, N, 2)
    dist_to_wp = torch.norm(diff, dim=-1)                        # (B, N)
    closest_idx = torch.argmin(dist_to_wp, dim=-1)                # (B,)

    N = waypoints_world.shape[0]
    next_idx = torch.clamp(closest_idx + 1, max=N - 1)

    wp_a = waypoints_world[closest_idx]  # (B, 2)
    wp_b = waypoints_world[next_idx]     # (B, 2)

    seg_vec = wp_b - wp_a                # (B, 2)
    robot_vec = robot_pos - wp_a         # (B, 2)

    # Project robot_vec onto seg_vec, clamp t to [0, 1]
    seg_len_sq = torch.sum(seg_vec * seg_vec, dim=-1) + 1e-8   # (B,)
    t = torch.sum(robot_vec * seg_vec, dim=-1) / seg_len_sq    # (B,)
    t = torch.clamp(t, 0.0, 1.0)

    proj_point = wp_a + t.unsqueeze(-1) * seg_vec  # (B, 2)
    cross_track = torch.norm(robot_pos - proj_point, dim=-1)   # (B,)

    return closest_idx, cross_track


# ===================== Subgoal selection =====================

def get_current_subgoal(
    robot_pos: torch.Tensor,
    path: list[tuple[float, float]] | None,
    final_target: torch.Tensor,
    env_origin: torch.Tensor,
    look_ahead_dist: float,
    device: torch.device,
    cached_closest_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a subgoal by walking *look_ahead_dist* metres along the path.

    This is distance-based (not count-based), so in tight corners with dense
    waypoints the subgoal stays close, preventing the robot from cutting corners.

    Args:
        robot_pos: (B, 2) world XY of each robot.
        path: List of (x, y) env-local waypoints.
        final_target: (B, 2) world XY of the final navigation target.
        env_origin: (2,) world XY of environment 0.
        look_ahead_dist: Distance [m] to walk forward along the path from projection.
        device: Target torch device.
        cached_closest_idx: If already computed this step, reuse it.

    Returns:
        subgoals: (B, 2) world XY of selected subgoal per robot.
        closest_idx: (B,) int32 — projected waypoint index (for caller reuse).
    """
    if not has_valid_path(path):
        return final_target, torch.zeros(
            robot_pos.shape[0], dtype=torch.int32, device=device
        )

    waypoints_world = waypoints_local_to_world(path, env_origin, device)  # (N, 2)

    if cached_closest_idx is None:
        cached_closest_idx, _ = project_onto_path(robot_pos, waypoints_world)

    N = waypoints_world.shape[0]

    # Ensure subgoal is always at least 1 step ahead of the projected point
    subgoal_idx = torch.clamp(cached_closest_idx + 1, max=N - 1)

    # If look_ahead_dist > 0, walk further along the path by distance
    if look_ahead_dist > 0:
        remaining = torch.full_like(cached_closest_idx, look_ahead_dist, dtype=torch.float32)
        for _ in range(N - 1):
            dist = torch.norm(
                waypoints_world[torch.clamp(subgoal_idx + 1, max=N - 1)]
                - waypoints_world[subgoal_idx],
                dim=-1,
            )
            advance = (remaining > 0) & (subgoal_idx + 1 < N)
            subgoal_idx = torch.where(advance, subgoal_idx + 1, subgoal_idx)
            remaining = remaining - torch.where(advance, dist, torch.zeros_like(dist))

    subgoals = waypoints_world[subgoal_idx]  # (B, 2)

    # When subgoal falls on the last waypoint, use the final target
    at_last = subgoal_idx >= N - 1
    subgoals[at_last] = final_target[at_last]

    return subgoals, cached_closest_idx


# ===================== Waypoint observation =====================

def get_waypoint_obs(
    robot_pos: torch.Tensor,
    path: list[tuple[float, float]] | None,
    env_origin: torch.Tensor,
    device: torch.device,
    cached_closest_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the waypoint observation vector: next 2 waypoints in robot frame.

    Returns (dx1, dy1, dx2, dy2) for each robot.

    Args:
        robot_pos: (B, 2) world XY of each robot.
        path: List of (x, y) env-local waypoints.
        env_origin: (2,) world XY of environment 0.
        device: Target torch device.
        cached_closest_idx: If already computed this step, reuse it.

    Returns:
        waypoint_obs: (B, 4) tensor [dx1, dy1, dx2, dy2].
        closest_idx: (B,) int32 — projected waypoint index.
    """
    B = robot_pos.shape[0]
    result = torch.zeros(B, 4, device=device)

    if not has_valid_path(path):
        return result, torch.zeros(B, dtype=torch.int32, device=device)

    waypoints_world = waypoints_local_to_world(path, env_origin, device)  # (N, 2)

    if cached_closest_idx is None:
        cached_closest_idx, _ = project_onto_path(robot_pos, waypoints_world)

    N = waypoints_world.shape[0]

    for j in range(2):
        wp_idx = torch.clamp(cached_closest_idx + 1 + j, max=N - 1)
        wp_pos = waypoints_world[wp_idx]  # (B, 2)
        result[:, j * 2 + 0] = wp_pos[:, 0] - robot_pos[:, 0]
        result[:, j * 2 + 1] = wp_pos[:, 1] - robot_pos[:, 1]

    return result, cached_closest_idx


# ===================== Replan trigger =====================

def check_replan_needed(
    cross_track: torch.Tensor,
    replan_cooldown: torch.Tensor,
    replan_threshold: float,
) -> torch.Tensor:
    """Return a boolean mask of robots that need path replanning.

    A robot needs replanning when its cross-track error exceeds
    *replan_threshold* AND its per-robot cooldown counter is zero.

    Args:
        cross_track: (B,) per-robot cross-track error [m].
        replan_cooldown: (B,) int32 cooldown counters.
        replan_threshold: Distance threshold [m].

    Returns:
        need_replan: (B,) bool tensor.
    """
    return (cross_track > replan_threshold) & (replan_cooldown == 0)


def decrement_cooldowns(replan_cooldown: torch.Tensor) -> torch.Tensor:
    """Decrement all positive cooldown counters by 1 (floor at 0).

    Args:
        replan_cooldown: (B,) int32 tensor.

    Returns:
        Updated cooldown tensor.
    """
    return torch.where(
        replan_cooldown > 0,
        replan_cooldown - 1,
        torch.zeros_like(replan_cooldown),
    )
