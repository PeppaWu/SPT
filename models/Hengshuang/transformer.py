from pointnet_util import index_points, square_distance, build_spike_node
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class _NeighborWeightedSum(torch.autograd.Function):
    """Weighted sum over point neighbors."""

    @staticmethod
    def forward(ctx, attn, values):
        ctx.save_for_backward(attn, values)
        return torch.einsum('bcnm,bcnm->bcn', attn, values)

    @staticmethod
    def backward(ctx, grad_output):
        attn, values = ctx.saved_tensors
        grad_output = grad_output.unsqueeze(-1)
        return grad_output * values, grad_output * attn


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k, timestep, spike_mode, use_encoder) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(
            build_spike_node(timestep, ['lif', 'elif', 'plif', 'if'], d_points) if spike_mode is not None else  nn.Identity(), # Mixer Lif/ HD-IF
            nn.Conv1d(d_points, d_model, 1), 
            nn.BatchNorm1d(d_model),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.Identity(),
        )
        self.fc2 = nn.Sequential(
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.Identity(),
            nn.Conv1d(d_model, d_points, 1), 
            nn.BatchNorm1d(d_points)
        )
        self.fc_delta = nn.Sequential(
            nn.Conv2d(3, d_model, 1),
            nn.BatchNorm2d(d_model),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.ReLU(),
            nn.Conv2d(d_model, d_model, 1),
            nn.BatchNorm2d(d_model),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.Identity(),
        )
        self.fc_gamma = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.BatchNorm2d(d_model),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.ReLU(),
            nn.Conv2d(d_model, d_model, 1),
            nn.BatchNorm2d(d_model),
        )

        self.w_qs = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1), 
            nn.BatchNorm1d(d_model),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.Identity(),
        )
        self.w_ks = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1), 
            nn.BatchNorm1d(d_model),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.Identity(),
        )
        self.w_vs = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1), 
            nn.BatchNorm1d(d_model),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.Identity(),
        )
        self.k = k
        self.use_encoder = use_encoder
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        T = xyz.shape[0]
        loc = xyz[0] if not self.use_encoder else xyz
        dists = square_distance(loc, loc)
        knn_idx = dists.argsort()[:, :, :self.k] \
                if not self.use_encoder else \
                dists.argsort()[:, :, :, :self.k]
        knn_xyz = index_points(loc, knn_idx)
        knn_idx = knn_idx.repeat(T, 1, 1, 1).flatten(0,1) \
                if not self.use_encoder else \
                knn_idx.flatten(0,1)

        features = features.flatten(0,1).permute(0,2,1).contiguous()
        pre = features

        x = self.fc1(features)

        q, k, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)
        k = index_points(k.permute(0,2,1), knn_idx).permute(0,3,1,2).contiguous()
        v = index_points(v.permute(0,2,1), knn_idx).permute(0,3,1,2).contiguous()
        
        pos_enc = self.fc_delta((xyz[:, :, :, None] - (knn_xyz.repeat(T, 1, 1, 1 ,1) \
                                                       if not self.use_encoder else knn_xyz)).flatten(0,1).permute(0,3,1,2).contiguous())  # T*B, C, N, M

        # broadcast mechanism
        attn = self.fc_gamma(q[:, :, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(1)), dim=-1)  # T*B, C, N, M
        
        res = _NeighborWeightedSum.apply(attn, v + pos_enc)
        res = self.fc2(res) + pre
        res = res.permute(0,2,1).reshape(T, xyz.shape[1], xyz.shape[2], -1)
        return res, attn
    
