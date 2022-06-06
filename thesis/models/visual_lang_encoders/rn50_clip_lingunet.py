from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.models.core.resnet import IdentityBlock, ConvBlock
from thesis.models.core.unet import Up
from thesis.models.core import fusion
from thesis.models.visual_lang_encoders.base_lingunet import BaseLingunet

from thesis.utils.utils import calc_cnn_out_size


class LangFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 fusion_input_dim, proj_input_dim, device,
                 up_factor=2, bilinear=True, lang_fusion_type='mult'):
        super().__init__()
        _fusion_fnc = fusion.names[lang_fusion_type]

        self.lang_fuser = _fusion_fnc(input_dim=fusion_input_dim).to(device)
        self.lang_proj = nn.Linear(proj_input_dim, out_channels).to(device)
        self.up_conv = Up(in_channels, out_channels // up_factor, bilinear).to(device)

    def forward(self, inp, skip_conn, l_input, x2_mask=None):
        x = self.lang_fuser(inp, l_input, x2_mask=x2_mask, x2_proj=self.lang_proj)
        x = self.up_conv(x, skip_conn)
        return x

class CLIPLingUNet(BaseLingunet):
    """ CLIP RN50 with U-Net skip connections """

    def __init__(self, input_shape, output_dim, cfg, device, clip_rn50):
        super(CLIPLingUNet, self).__init__(input_shape, output_dim, cfg, device)
        self.output_dim = output_dim
        self.input_dim = 2048  # penultimate layer channel-size of CLIP-RN50
        self.batchnorm = self.cfg['batchnorm']
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1
        self.clip_rn50 = clip_rn50
        self._build_decoder()
    
    def calc_img_enc_size(self):
        test_tensor = torch.zeros(self.input_shape).permute(2, 0, 1)
        test_tensor = test_tensor.to(self.device).unsqueeze(0)
        shape = self.encode_image(test_tensor)[0].shape[1:]
        return shape

    def _build_decoder(self):
        # language
        self.proj_input_dim = 1024
        out_channels = 1024
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        _enc_im_shape = self.calc_img_enc_size()
        h = calc_cnn_out_size(_enc_im_shape[1], k=3, s=1, p=1)
        layers_info = [{"name": "encoder_out", "shape": _enc_im_shape},
                        {"name": "conv1",
                        "shape": (out_channels, h, h)}]

        # Decoder layers w/language conditioning
        _lang_blocks = []
        in_channels = out_channels * 2
        for i in range(1, 4):
            out_channels = in_channels // 2
            lang_block = LangFusionBlock(in_channels,
                                         out_channels,
                                         self.input_dim // (2 * i),
                                         self.proj_input_dim,
                                         self.device,
                                         self.up_factor,
                                         self.bilinear,
                                         self.lang_fusion_type)
            in_channels = out_channels
            _lang_blocks.append(lang_block)

            # Save shape
            h = layers_info[-1]["shape"][-1] * 2
            h = calc_cnn_out_size(h, k=3, p=1)
            h = calc_cnn_out_size(h, k=3, p=1)
            layers_info.append({"name": "conv_lang%d" % i,
                                "shape": (out_channels // self.up_factor, h, h)})
        self.lang_blocks = nn.ModuleList(_lang_blocks)

        # Decoder layers w/o language
        _decoder_blocks = []
        in_channels = 128
        h = self.input_shape[0] // 2
        for i in range(1, 4):
            out_channels = in_channels // 2
            out_c = [out_channels, out_channels, out_channels]
            layer = nn.Sequential(
                ConvBlock(in_channels, out_c, kernel_size=3, stride=1, batchnorm=self.batchnorm),
                IdentityBlock(out_channels, out_c, kernel_size=3, stride=1, batchnorm=self.batchnorm),nn.UpsamplingBilinear2d(scale_factor=2)).to(self.device)
            _decoder_blocks.append(layer)
            in_channels = out_channels

            # Calc out shape
            layers_info.append({"name": "decoder%d" % i,
                                "shape": (out_channels, h, h)})
            h = layers_info[-1]["shape"][-1] * 2

        self.decoder_blocks = nn.ModuleList(_decoder_blocks)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, self.output_dim, kernel_size=1)
        )
        h = calc_cnn_out_size(h, k=1)
        layers_info.append({"name": "conv2",
                            "shape": (self.output_dim, h, h)})
        self.decoder_layers = layers_info

    def encode_image(self, img):
        with torch.no_grad():
            img_encoding, img_im = self.clip_rn50.visual.prepool_im(img)
        return img_encoding, img_im

    def forward(self, x, text_enc):
        in_type = x.dtype
        in_shape = x.shape
        x = x[:,:3]  # select RGB
        x, im = self.encode_image(x)
        x = x.to(in_type)

        # encoded text
        l_enc, l_emb, l_mask  = text_enc
        l_input = l_enc.to(dtype=x.dtype)
    
        _info = {"hidden_layers": [x],
                 "text_enc": l_input,
                 "fusion_type": self.lang_fusion_type}
        # Decoder
        # encode image
        assert x.shape[1] == self.input_dim
        x = self.conv1(x)

        for i, lang_block in enumerate(self.lang_blocks, 2):
            x = lang_block(x, im[-i], l_input, x2_mask=l_mask)
            _info["hidden_layers"].append(x)

        for layer in self.decoder_blocks:
            x = layer(x)
            _info['hidden_layers'].append(x)

        x = self.conv2(x)
        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        return x, _info