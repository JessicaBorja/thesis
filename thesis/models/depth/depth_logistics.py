import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.models.core.clip import build_model, load_clip, tokenize
from thesis.models.core import fusion
from torch.distributions import Normal, Independent
import torchvision.models as models

import numpy as np
from thesis.utils.utils import calc_cnn_out_size


class DepthEstimation(nn.Module):
    """ CLIP RN50 with U-Net skip connections """

    def __init__(self, input_shape, output_dim, cfg, device):
        super(DepthEstimation, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.input_dim = 2048  # penultimate layer channel-size of CLIP-RN50
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['batchnorm']
        self.lang_fusion_type = self.cfg['lang_fusion_type']
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1

        self.one_hot_embedding_eye = torch.eye(self.n_dist)
        self.action_max_bound = torch.tensor([4.0])
        self.action_min_bound = torch.tensor([2.0])
        self.n_dist = 100 # cfg.depth_resolution
        self.img_encoder = self._load_img_encoder()
        self._build_decoder()
    
    def calc_img_enc_size(self):
        test_tensor = torch.zeros(self.input_shape).permute(2, 0, 1)
        test_tensor = test_tensor.to(self.device).unsqueeze(0)
        shape = self.encode_image(test_tensor)[0].shape[1:]
        return shape

    def _load_img_encoder(self):
        model = models.resnet18(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        return model

    def sample(self, dist, reparametrize=True):
        logit_probs, log_scales, means = dist

        # Selecting Logistic distribution (Gumbel Sample)
        r1, r2 = 1e-5, 1.0 - 1e-5
        temp = (r1 - r2) * torch.rand(means.shape, device=means.device) + r2
        temp = logit_probs - torch.log(-torch.log(temp))
        argmax = torch.argmax(temp, -1)

        # TODO: find out why mypy complains about type
        dist = self.one_hot_embedding_eye[argmax]

        # Select scales and means
        log_scales = (dist * log_scales).sum(dim=-1)
        means = (dist * means).sum(dim=-1)

        # Inversion sampling for logistic mixture sampling
        scales = torch.exp(log_scales)  # Make positive
        u = (r1 - r2) * torch.rand(means.shape, device=means.device) + r2
        sample = means + scales * (torch.log(u) - torch.log(1.0 - u))
        return sample

    def encode_image(self, img):
        return self.img_encoder(img)

    def _build_decoder(self):
        # B, C, H, W
        self.proj_input_dim = 512 if 'word' in self.lang_fusion_type else 1024
        _test_tensor = torch.zeros(self.input_shape).permute((2, 0, 1)).unsqueeze(0)
        shape = self.encode_image(_test_tensor).shape

        linear_in = np.prod(shape)
        hidden_size = 256
        self.fc1 = nn.Linear(linear_in, self.proj_input_dim)
        self.fc2 = nn.Linear(self.proj_input_dim, hidden_size)

        self.prob_fc = nn.Linear(hidden_size, self.n_dist)
        self.mean_fc = nn.Linear(hidden_size, self.n_dist)
        self.log_scale_fc = nn.Linear(hidden_size, self.n_dist)

    def loss(self, pred, gt_depth):
        logit_probs, log_scales, means = pred

        # Appropriate scale
        log_scales = torch.clamp(log_scales, min=-7.0)
        
        # Broadcast actions (B, 1, N_DIST)
        gt_depth = gt_depth.unsqueeze(-1) * torch.ones(1, 1, self.n_dist)

        # Approximation of CDF derivative (PDF)
        centered_actions = gt_depth - means
        inv_stdv = torch.exp(-log_scales)

        assert torch.is_tensor(self.action_max_bound)
        assert torch.is_tensor(self.action_min_bound)

        act_range = (self.action_max_bound - self.action_min_bound) / 2.0
        plus_in = inv_stdv * (centered_actions + act_range / (self.num_classes - 1))
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_actions - act_range / (self.num_classes - 1))
        cdf_min = torch.sigmoid(min_in)

        # Corner Cases
        log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
        log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)
        # Log probability in the center of the bin
        mid_in = inv_stdv * centered_actions
        log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
        # Probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        # Log probability
        log_probs = torch.where(
            gt_depth < self.action_min_bound + 1e-3,
            log_cdf_plus,
            torch.where(
                gt_depth > self.action_max_bound - 1e-3,
                log_one_minus_cdf_min,
                torch.where(
                    cdf_delta > 1e-5,
                    torch.log(torch.clamp(cdf_delta, min=1e-12)),
                    log_pdf_mid - np.log((self.num_classes - 1) / 2),
                ),
            ),
        )
        log_probs = log_probs + F.log_softmax(logit_probs, dim=-1)
        loss = -torch.sum(self.log_sum_exp(log_probs), dim=-1).mean()
        return loss

    def log_sum_exp(self, x):
        """numerically stable log_sum_exp implementation that prevents overflow"""
        axis = len(x.size()) - 1
        m, _ = torch.max(x, dim=axis)
        m2, _ = torch.max(x, dim=axis, keepdim=True)
        return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

    def forward(self, x, l_enc):
        in_type = x.dtype
        x = x[:,:3]  # select RGB
        x = self.encode_image(x)
        x = x.to(in_type)

        # encode text
        l_enc, _, l_mask = l_enc
        l_input = l_enc.to(dtype=x.dtype)
    
        _info = {"hidden_layers": [x],
                 "l_mask": l_mask,
                 "l_input": l_input,
                 "fusion_type": self.lang_fusion_type}
        # Decoder
        B, C, H, W = x.shape
        x = x.reshape((B, -1))
        x = F.relu(self.fc1(x))
        x = x * l_input
        x = F.relu(self.fc2(x))

        probs = self.prob_fc(x)
        means = self.mean_fc(x)
        log_scales = self.log_scale_fc(x)
        # log_scales = torch.clamp(log_scales, min=self.log_scale_min)

        # Appropriate dimensions
        logit_probs = probs.view(B, 1, self.n_dist)
        means = means.view(B, 1, self.n_dist)
        log_scales = log_scales.view(B, 1, self.n_dist)
        dist = (logit_probs, log_scales, means)
    
        return dist, _info