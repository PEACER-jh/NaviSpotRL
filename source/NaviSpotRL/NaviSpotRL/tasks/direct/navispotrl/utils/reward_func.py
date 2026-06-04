"""Reward — v63: LSTM-friendly simplified obstacle avoidance.

With LSTM temporal understanding, we remove the manual heuristics
(episode warmup, approach-gate) and let the network learn when to
avoid vs when to approach. Cleaner reward signal = faster learning.
"""

from __future__ import annotations

import torch

REACH_THRESHOLD = 0.3
OBSTACLE_HARD_LIMIT = 0.3    # v61: only penalize when collision imminent
OBSTACLE_SOFT_LIMIT = 0.8    # v61: gentle warning zone, much narrower
OBSTACLE_HARD_WEIGHT = 10.0  # v61: strong punch right before collision
OBSTACLE_SOFT_WEIGHT = 0.3   # v61: light touch


def distance_progress_reward(
    prev_dist: torch.Tensor,
    curr_dist: torch.Tensor,
    alignment: torch.Tensor,
    v_max: float,
    dt: float,
) -> torch.Tensor:
    # v61: stronger drive toward target
    delta = prev_dist - curr_dist
    return torch.where(delta > 0, delta * 12.0, delta * 4.0)


def steer_toward_target_reward(
    cross: torch.Tensor, ang_vel: torch.Tensor, alignment: torch.Tensor
) -> torch.Tensor:
    misalign = (1.0 - alignment)
    weight = misalign * misalign + 0.35
    return cross * ang_vel * weight * 8.0


def speed_reward(fwd_vel: torch.Tensor, alignment: torch.Tensor) -> torch.Tensor:
    # v61+: higher baseline so moving forward beats turning at alignment
    baseline = fwd_vel * 0.8
    gate = alignment * alignment
    return baseline + fwd_vel * 2.0 * gate


def misalign_speed_penalty(fwd_vel: torch.Tensor, alignment: torch.Tensor) -> torch.Tensor:
    misalign = 1.0 - alignment * alignment
    return -fwd_vel * 1.5 * misalign


def obstacle_avoid_reward(lidar_rays: torch.Tensor) -> torch.Tensor:
    """v63: Clean sector-based obstacle penalty — no hand-crafted gates.

    lidar_rays: (B, 64) — flat: first 32 = front, last 32 = back.

    LSTM+Attention provides temporal/spatial understanding, so we remove:
      - Episode warmup (LSTM learns to leave spawn naturally)
      - Approach gate (LSTM knows if it's approaching or retreating)
      - Wide proximity gate (tighten to 1.5m — only penalize nearby)

    6 directional sectors give PPO gradient: "turning left reduces right-side penalty".
    Total penalty clamped to [-3.0, 0] for value loss stability.
    """
    B = lidar_rays.shape[0]
    device = lidar_rays.device

    # Proximity gate: >1.5m fully safe, <0.5m full penalty
    min_dist = lidar_rays.min(dim=-1).values  # (B,)
    proximity = torch.clamp((1.5 - min_dist) / 1.0, min=0.0, max=1.0)

    front = lidar_rays[:, :32]
    back  = lidar_rays[:, 32:]

    total = torch.zeros(B, device=device)

    sectors = [
        (front[:, 0:10],   "front_left",   0.6),
        (front[:, 10:22],  "front_center", 1.5),   # dead ahead most dangerous
        (front[:, 22:32],  "front_right",  0.6),
        (back[:, 0:10],    "back_left",    0.15),
        (back[:, 10:22],   "back_center",  0.2),
        (back[:, 22:32],   "back_right",   0.15),
    ]

    for rays, _label, weight in sectors:
        topk_vals, _ = torch.topk(-rays, k=min(3, rays.shape[-1]), dim=-1)
        sector_dist = -topk_vals.mean(dim=-1)

        soft_violation = torch.clamp(OBSTACLE_SOFT_LIMIT - sector_dist, min=0.0, max=1.0)
        soft_penalty = soft_violation * OBSTACLE_SOFT_WEIGHT

        hard_violation = torch.clamp(OBSTACLE_HARD_LIMIT - sector_dist, min=0.0)
        hard_penalty = hard_violation * OBSTACLE_HARD_WEIGHT

        total = total - (soft_penalty + hard_penalty) * weight

    total = torch.clamp(total, min=-3.0, max=0.0)
    return total * proximity


def alignment_reward(alignment: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(alignment)


def idle_penalty(fwd_vel: torch.Tensor) -> torch.Tensor:
    # v61: penalize standing still — -0.2 per step when not moving
    return torch.where(fwd_vel < 0.01, torch.tensor(-0.2, device=fwd_vel.device), torch.tensor(0.0, device=fwd_vel.device))


def side_drive_cost(fwd_vel: torch.Tensor, alignment: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(fwd_vel)


def yaw_damping(ang_vel: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(ang_vel)


# ---- Compatibility wrappers ----

def misalign_drive_penalty(fwd_vel, alignment):
    return misalign_speed_penalty(fwd_vel, alignment)

def yaw_stability_penalty(ang_vel, alignment):
    return yaw_damping(ang_vel)

def turn_guide_reward(cross, ang_vel, alignment):
    return steer_toward_target_reward(cross, ang_vel, alignment)

def turn_reward(cross, ang_vel, alignment):
    return steer_toward_target_reward(cross, ang_vel, alignment)

def aligned_forward_reward(fwd_vel, alignment):
    return speed_reward(fwd_vel, alignment)

def backward_drive_penalty(fwd_vel):
    return torch.zeros_like(fwd_vel)

def waypoint_reach(distance):
    return (distance < REACH_THRESHOLD).float() * 5.0

def tilt_penalty(roll, pitch):
    return torch.zeros_like(roll)

def stuck_penalty(stuck_mask):
    return torch.zeros_like(stuck_mask)
