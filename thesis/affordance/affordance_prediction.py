import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import logging

import thesis.models as models
from thesis.models.lang_fusion.one_stream_attention_lang_fusion_pixel import AttentionLangFusionPixel
from thesis.utils.utils import add_img_text, tt, blend_imgs,get_transforms, resize_pixel, unravel_idx
from thesis.utils.losses import cross_entropy_with_logits


class AffPrediction(LightningModule):
    def __init__(self, cfg, in_shape=(200, 200, 3), transforms=None, *args, **kwargs):
        super().__init__()
        self.text_enc = hydra.utils.instantiate(cfg.text_encoder)
        self.affordance = hydra.utils.instantiate(cfg.affordance_predictor)
        self.depth = hydra.utils.instantiate(cfg.depth_predictor)

    def training_step(self, batch, batch_idx):
        aff_loss, aff_info = self.affordance.training_step()
        depth_loss, depth_info = self.depth.training_step()

        total_loss = aff_loss + depth_loss
        return total_loss

    def validation_step(self, batch, batch_idx):
        aff_loss, aff_info = self.affordance.validation_step()
        depth_loss, depth_info = self.depth.validation_step()

        return
