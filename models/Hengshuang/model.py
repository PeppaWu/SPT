import torch
import math
import globals
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction, build_spike_node, farthest_point_sample, index_points

from .transformer import TransformerBlock, build_spike_node

class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels, timestep, spike_mode, use_encoder):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], timestep, spike_mode, use_encoder, group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)

class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.dataset.num_class, cfg.input_dim
        blocks, num_samples = cfg.model.blocks, cfg.model.num_samples
        spike_mode, timestep, use_encoder = cfg.model.spike_mode, cfg.model.timestep, cfg.model.use_encoder
        assert len(blocks) == nblocks+1, "Block mismatches"

        self.fc1 = nn.Sequential(
            nn.Conv1d(d_points, 32, 1),
            nn.BatchNorm1d(32),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.ReLU(),
            nn.Conv1d(32, 32, 1), 
            nn.BatchNorm1d(32),
        )

        transblock = lambda channel: TransformerBlock(channel, cfg.model.transformer_dim, nneighbor, timestep, spike_mode, use_encoder)
        self.transformer1 = nn.ModuleList(transblock(32) for _ in range(blocks[0]))

        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel], timestep, spike_mode, use_encoder))
            for _ in range(blocks[i + 1]):
                self.transformers.append(transblock(channel))

        self.nblocks = nblocks
        self.blocks = blocks
    
    def forward(self, x):
        T, B, N, C = x.shape
        xyz = x[..., :3]
        x = self.fc1(x.flatten(0, 1).permute(0, 2, 1).contiguous())
        x = x.view(T, B, -1, N).permute(0, 1, 3, 2).contiguous()
        points = self.transformer1[0](xyz, x)[0]

        xyz_and_feats = [(xyz, points)]
        id = 0 
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            for _ in range(self.blocks[i + 1]):                
                points = self.transformers[id](xyz, points)[0]
                id += 1
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.dataset.num_class, cfg.input_dim
        spike_mode, timestep, use_encoder, num_samples = cfg.model.spike_mode, cfg.model.timestep, cfg.model.use_encoder, cfg.model.num_samples
        self.fc2 = nn.Sequential(
            build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.Identity(),
            nn.Conv1d(32 * 2 ** nblocks, 256, 1),
            nn.BatchNorm1d(256),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.ReLU(),
            nn.Conv1d(256, 64, 1),
            nn.BatchNorm1d(64),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.ReLU(),
            nn.Conv1d(64, n_c, 1),
        )
        self.nblocks = nblocks
        self.T = timestep
        self.spike_mode  = spike_mode
        self.use_encoder = use_encoder
        self.num_samples = max(npoints//self.T, num_samples)
    
    def random_SDE(self, x):
        B, N, C = x.shape
        assert N%self.T == 0, "timstep is invalid"
        x = x.view(B, self.T, N//self.T, C).transpose(0, 1)
        return x
    
    def scan_SDE(self, x):
        B, N, C = x.shape
        _, indices = torch.sort(x[:, :, 0], dim=1)
        x = torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, C))
        x = x.expand(math.ceil(self.num_samples*self.T/N), *x.shape).transpose(0,1)
        x = x.flatten(1,2)[:,:self.num_samples*self.T]
        scan = x.view(B, self.T, self.num_samples, C).transpose(0, 1)
        return scan

    def queue_SDE(self, x):
        def queue_mask(loc, fps_idx):
            # Remove sampled indices while preserving queue order.
            size = loc.shape[1]
            ids = torch.arange(size, device=loc.device).expand(B, -1).clone()
            ids.scatter_(1, fps_idx, size)
            remaining = ids.sort(dim=1).values[:, :size - fps_idx.shape[1]]
            return loc.gather(1, remaining.unsqueeze(-1).expand(-1, -1, C))
        
        B, N, C = x.shape
        loc = x
        npoint = self.num_samples
        res = (N - npoint)//(self.T-1) if self.T != 1 else 0

        onion = torch.zeros(self.T, B, npoint, C).to(x.device)
        fps_idx = farthest_point_sample(loc[..., :3], npoint)
        onion[0] = index_points(loc, fps_idx)
        loc = queue_mask(loc, fps_idx)    

        for i in range(1, self.T):
            if loc.shape[1] == 0: onion[i] = onion[i-1]
            else:
                fps_idx = farthest_point_sample(loc[..., :3], res)
                onion[i, :, :npoint - res] = onion[i - 1][:, res:]
                onion[i, :, npoint - res:] = index_points(loc, fps_idx)
                loc = queue_mask(loc, fps_idx)
        return onion

    def forward(self, x):
        # Convert to Spike
        assert len(x.shape) < 4, "shape of inputs is invalid"
        st = time.time()
        if self.spike_mode is not None:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1) \
                if not self.use_encoder else \
                self.queue_SDE(x)
        else:
            x = x.unsqueeze(0)
        end = time.time()
        globals.MID_TIME = end - st

        # Backbone
        points, _ = self.backbone(x)

        # Classification head.
        points = points.mean(2) if len(points.shape) == 4 else points.mean(1)
        points = points.unsqueeze(-1)
        res = self.fc2(points.flatten(0,1))
        res = res.view(self.T, -1, *res.shape[1:]).mean(0)
        return res.squeeze(2)

    
