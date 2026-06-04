"""Depth-map processing: convert camera depth images to pseudo-LiDAR rays.

Extracted from NavispotrlEnv._depth_to_rays.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def depth_to_rays(
    front_depth: torch.Tensor,
    back_depth: torch.Tensor,
    n_rays: int,
    camera_height: float = 0.11,
    vfov_deg: float = 17.0,
    clip_max: float = 3.0,
) -> torch.Tensor:
    """Extract obstacle-line rays from front+back depth maps.

    Pipeline aligned with Rerun depth visualization:
      1. Clip to [0.05, clip_max] m (same range as Rerun display)
      2. Per-pixel ground filtering via physical model
      3. Per-column min over non-ground pixels
      4. Adaptive min-pool downsample to ``n_rays // 2`` rays per camera

    Args:
        front_depth: (B, H, W, 1) tensor — raw depth in **metres**.
        back_depth:  (B, H, W, 1) tensor — raw depth in **metres**.
        n_rays: Total number of output rays (front + back).
        camera_height: Height of camera above ground [m].
        vfov_deg: Vertical field of view [degrees].
        clip_max: Maximum depth to consider [m] (matches Rerun clip).

    Returns:
        lidar_rays: (B, n_rays) flat tensor — obstacle distances in metres.
    """
    B, H, W, _ = front_depth.shape

    # Step 0: clip to Rerun-aligned range
    front = torch.clamp(front_depth, 0.05, clip_max)  # (B, H, W, 1)
    back = torch.clamp(back_depth, 0.05, clip_max)

    # Step 1: compute per-row theoretical ground distance
    vfov_rad = math.radians(vfov_deg)
    # Row angles: top (+vfov/2) → bottom (-vfov/2)
    row_angles = torch.linspace(vfov_rad / 2, -vfov_rad / 2, H, device=front_depth.device)
    # Ground intersection along the camera ray: dist = h / sin(|angle|) for angle < 0
    sin_angle = torch.sin(torch.abs(row_angles)) + 1e-8
    ground_dists = camera_height / sin_angle  # (H,)
    # Rows at or above horizontal never see ground → set to clip_max (ignored)
    ground_dists[row_angles >= 0] = clip_max

    # Step 2: per-pixel ground mask
    # A pixel is "ground" if its depth is within threshold of theoretical ground
    ground_threshold = 0.3  # ±30 cm tolerance
    is_ground = (front.squeeze(-1) - ground_dists.view(1, H, 1)).abs() < ground_threshold
    is_ground_back = (back.squeeze(-1) - ground_dists.view(1, H, 1)).abs() < ground_threshold

    # Step 3: mask ground → clip_max (won't affect min), then per-column min
    front_masked = front.squeeze(-1).clone()
    back_masked = back.squeeze(-1).clone()
    front_masked[is_ground] = clip_max
    back_masked[is_ground_back] = clip_max

    front_line = front_masked.min(dim=1).values  # (B, W)
    back_line = back_masked.min(dim=1).values

    # Step 4: adaptive min-pool downsample (preserves thin obstacles)
    target_w = n_rays // 2
    front_rays = -F.adaptive_max_pool1d(-front_line.unsqueeze(1), target_w).squeeze(1)
    back_rays = -F.adaptive_max_pool1d(-back_line.unsqueeze(1), target_w).squeeze(1)

    # Step 5: stack → (B, 2, n_rays//2), then flatten → (B, n_rays)
    lidar_rays = torch.stack([front_rays, back_rays], dim=1)  # (B, 2, N//2)
    lidar_rays = lidar_rays.reshape(B, -1)                    # (B, n_rays)
    return lidar_rays
