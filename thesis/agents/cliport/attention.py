import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cliport.models as models
from cliport.utils import utils


class OneStreamAttentionLangFusion(nn.Module):
    """Attention (a.k.a Pick) module with language features fused at the bottleneck."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        super().__init__()
        self.fusion_type = cfg.train.attn_stream_fusion_type
        self.stream_fcn = stream_fcn
        self.n_rotations = n_rotations
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg.train.batchnorm

        self.padding = np.zeros((3, 2), dtype=int) # H, W, C
        max_dim = np.max(in_shape[:2])
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)  # H, W, C 

        in_shape = np.array(in_shape)
        in_shape += np.sum(self.padding, axis=1)
        in_shape = list(in_shape)

        # for torch: left, right,(W) top, bottom,(H) front, back(C)
        self.padding = self.padding[[1, 0, 2]] # C, H, W
        self.padding = tuple(self.padding.flatten())
        self.in_shape = in_shape

        self.rotator = utils.ImageRotator(self.n_rotations)

        self._build_nets()

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        print(f"Attn FCN: {stream_one_fcn}")

    def attend(self, x, l):
        x = self.attn_stream_one(x, l)
        return x

    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        in_data = F.pad(inp_img, self.padding, mode='constant')
        in_tens = in_data.to(dtype=torch.float, device=self.device) # [B 3 H W]
        # Rotation pivot.
        pv = np.array(in_tens.shape[2:]) // 2

        # Rotate input.
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)

        # Forward pass.
        logits = []
        for x, lang_goal in zip(in_tens, lang_goal):
            lgts = self.attend(x, lang_goal)
            logits.append(lgts)
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = np.array([self.padding[2], self.padding[0]])  # top(H), left(W)
        c1 = c0 + inp_img.shape[2:]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]

        logits = logits.permute(1, 2, 3, 0)  # [B H W 1]
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output