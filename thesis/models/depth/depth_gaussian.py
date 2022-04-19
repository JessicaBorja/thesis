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

        # Use clip preprocessing
        self.loss_fcn = nn.GaussianNLLLoss()
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

    def sample(self, depth_dist, reparametrize=True):
        dist, _, _ = depth_dist
        if reparametrize:
            sample = dist.rsample()
        else:
            sample = dist.sample()
        return sample

    def encode_image(self, img):
        return self.img_encoder(img)

    def _build_decoder(self):
        # B, C, H, W
        self.proj_input_dim = 512 if 'word' in self.lang_fusion_type else 1024
        _test_tensor = torch.zeros(self.input_shape).permute((2, 0, 1)).unsqueeze(0)
        shape = self.encode_image(_test_tensor).shape

        linear_in = np.prod(shape)
        self.fc1 = nn.Linear(linear_in, self.proj_input_dim)
        self.fc2 = nn.Linear(self.proj_input_dim, 256)
        self.depth_mu = nn.Linear(256, 1)
        self.depth_sigma = nn.Linear(256, 1)

    def depth_forward(self, x, l_input):
        x = F.relu(self.fc1(x))
        x = x * l_input
        x = F.relu(self.fc2(x))
        mu = self.depth_mu(x)
        log_sigma = self.depth_sigma(x)
        # avoid log_sigma to go to infinity
        sigma = torch.clamp(log_sigma, -20, 2).exp()

        # Sample
        dist = Independent(Normal(mu, sigma), 1)
        return dist, mu, sigma

    def loss(self, pred, gt_depth):
        dist, mu, sigma = pred['depth_dist']
        depth_loss = self.loss_fcn(mu, gt_depth, sigma)
        # depth_loss = -pred["depth_dist"].log_prob(gt_depth).mean()
        # neg_samples = pred["depth_dist"].sample()
        # depth_loss += pred["depth_dist"].log_prob(neg_samples).mean()
        # pred_depth = out["depth_dist"].rsample()
        # depth_loss = F.mse_loss(pred_depth, gt_depth)
        return depth_loss

    def forward(self, x, l_enc):
        in_type = x.dtype
        x = x[:,:3]  # select RGB
        x = self.encode_image(x)
        x = x.to(in_type)

        # encode text
        l_enc, l_emb, l_mask = l_enc
        l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=x.dtype)
    
        _info = {"hidden_layers": [x],
                 "l_mask": l_mask,
                 "l_input": l_input,
                 "fusion_type": self.lang_fusion_type}
        # Decoder
        B, C, H, W = x.shape
        x = x.reshape((B, -1))
        dist = self.depth_forward(x, l_input)

        return dist, _info