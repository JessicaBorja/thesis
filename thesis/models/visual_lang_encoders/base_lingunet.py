import torch
import torch.nn as nn

class BaseLingunet(nn.Module):
    """BaseClass with U-Net skip connections and [] language encoder"""
    def __init__(self, input_shape, output_dim, cfg, device, *args, **kwargs):
        super().__init__()
        self.input_shape = input_shape
        self.cfg = cfg
        self.device = device
        self.lang_fusion_type = self.cfg['lang_fusion_type']