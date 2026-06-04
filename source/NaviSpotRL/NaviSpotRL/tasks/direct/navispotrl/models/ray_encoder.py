"""RayEncoder — pose-conditioned cross-attention over lidar rays.

Inspired by GRALP's RayEncoder:
  1D Conv + Depthwise Separable + SqueezeExcite → ray features
  Learnable queries + pose embedding → cross-attend to ray features
  Output: spatial understanding of obstacles relative to robot state.

Architecture (slim version):
  lidar_rays (B, 64) → Conv1D(1→32) → DW-Sep(3 layers, dil=[1,2,4]) → SE
                      ↓                                        ↓
                    to_k, to_v (K, V)                      pose_mlp
                                                              ↓
                                              Cross-Attention(Q_pose, K, V)
                                                              ↓
                                            [z_mean | gavg | q_mean] → 128D
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== Utilities ====================

def _circular_pad1d(x: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return x
    left = x[..., -pad:]
    right = x[..., :pad]
    return torch.cat([left, x, right], dim=-1)


# ==================== Squeeze-and-Excitation (1D) ====================

class SqueezeExcite1D(nn.Module):
    """Channel-wise attention: learn which feature channels matter most."""
    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        hid = max(8, ch // r)
        self.fc1 = nn.Linear(ch, hid)
        self.fc2 = nn.Linear(hid, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


# ==================== Depthwise Separable Conv1D ====================

class DepthwiseSeparable1D(nn.Module):
    """Depthwise + Pointwise conv with GELU + BatchNorm."""
    def __init__(self, ch: int, kernel: int = 5, dilation: int = 1):
        super().__init__()
        self.kernel = int(kernel)
        self.dil = int(dilation)
        self.dw = nn.Conv1d(ch, ch, kernel_size=kernel, groups=ch, bias=False, dilation=self.dil)
        self.pw = nn.Conv1d(ch, ch, kernel_size=1)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = ((self.kernel - 1) * self.dil) // 2
        if pad > 0:
            x = _circular_pad1d(x, pad)
        out = self.dw(x)
        out = F.gelu(out)
        out = self.pw(out)
        out = self.bn(out)
        return out


# ==================== RayBranch ====================

class RayBranch(nn.Module):
    """1D Conv backbone for lidar rays.

    lidar_rays (B, 64) → expand(32) → DW-Sep × 3 (dil=1,2,4) + SE → (B, 32, 64)
    """
    def __init__(self, in_ch: int = 1, hidden: int = 32, layers: int = 3, kernel: int = 5):
        super().__init__()
        self.in_ch = int(in_ch)
        self.expand = nn.Conv1d(self.in_ch, hidden, kernel_size=1)
        dilations = [1, 2, 4][:layers]
        blocks = []
        for d in dilations:
            blocks += [
                DepthwiseSeparable1D(hidden, kernel=kernel, dilation=d),
                nn.GELU(),
                SqueezeExcite1D(hidden, r=4),
            ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 64) → (B, 1, 64)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.expand(x)      # (B, hidden, 64)
        x = self.blocks(x)      # (B, hidden, 64)
        return x


# ==================== RayEncoder (with Cross-Attention) ====================

class RayEncoder(nn.Module):
    """Pose-conditioned cross-attention over lidar ray features.

    Args:
        ray_dim: number of lidar rays (64).
        pose_dim: dimension of pose condition (9: policy+waypoints).
        hidden: RayBranch hidden channels.
        d_model: attention dimension.
        num_queries: number of learnable query vectors.
        num_heads: attention heads.
        output_dim: final output dimension.
    """
    def __init__(
        self,
        ray_dim: int = 64,
        pose_dim: int = 9,
        hidden: int = 32,
        d_model: int = 64,
        num_queries: int = 4,
        num_heads: int = 4,
        output_dim: int = 128,
    ):
        super().__init__()
        self.ray_dim = int(ray_dim)
        self.pose_dim = int(pose_dim)
        self.d_model = int(d_model)
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        # Ray feature extraction
        self.br_obs = RayBranch(in_ch=1, hidden=hidden, layers=3)
        self.to_k = nn.Conv1d(hidden, d_model, kernel_size=1)
        self.to_v = nn.Conv1d(hidden, d_model, kernel_size=1)

        # Pose encoder → query generator
        self.pose_mlp = nn.Sequential(
            nn.Linear(self.pose_dim, d_model),
            nn.ReLU(),
        )

        # Learnable query vectors
        init_scale = 1.0 / math.sqrt(max(1, d_model))
        self.q_params = nn.Parameter(torch.randn(self.num_queries, d_model) * init_scale)

        # Output projection
        self.post = nn.Sequential(
            nn.Linear(d_model * 3, output_dim), nn.ReLU(),
            nn.Linear(output_dim, output_dim), nn.ReLU(),
        )

    def forward(self, rays: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rays: (B, 64) lidar ray distances.
            pose: (B, pose_dim) robot state for attention query.
        Returns:
            spatial_feat: (B, output_dim).
        """
        B = rays.shape[0]

        # 1. Extract ray features
        Fmap = self.br_obs(rays)          # (B, hidden, 64)
        K = self.to_k(Fmap).transpose(1, 2)  # (B, 64, d_model)
        V = self.to_v(Fmap).transpose(1, 2)  # (B, 64, d_model)

        # 2. Pose → query
        q_pose = self.pose_mlp(pose)                     # (B, d_model)
        q = self.q_params.unsqueeze(0) + q_pose.unsqueeze(1)  # (B, num_queries, d_model)

        # 3. Multi-head Cross-Attention
        H = self.num_heads
        Dh = self.d_model // H
        K_h = K.view(B, K.shape[1], H, Dh)  # (B, 64, H, Dh)
        V_h = V.view(B, V.shape[1], H, Dh)
        Q_h = q.view(B, self.num_queries, H, Dh)

        attn_logits = torch.einsum('bmhd,bnhd->bmhn', Q_h, K_h) / math.sqrt(Dh)
        attn = torch.softmax(attn_logits, dim=-1)       # (B, M, H, N)
        z_h = torch.einsum('bmhn,bnhd->bmhd', attn, V_h)  # (B, M, H, Dh)
        z = z_h.reshape(B, self.num_queries, self.d_model)  # (B, M, d_model)

        # 4. Pool and concat
        z_mean = z.mean(dim=1)          # (B, d_model)
        q_mean = q.mean(dim=1)          # (B, d_model)
        gavg = V.mean(dim=1)            # (B, d_model) — global average of all rays

        g = torch.cat([z_mean, gavg, q_mean], dim=-1)  # (B, 3*d_model)
        g = self.post(g)                                # (B, output_dim)
        return g
