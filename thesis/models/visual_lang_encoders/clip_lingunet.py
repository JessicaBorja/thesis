import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.models.core.resnet import IdentityBlock, ConvBlock
from thesis.models.core.unet import Up
from thesis.models.core.clip import build_model, load_clip, tokenize
from thesis.models.core import fusion

from thesis.utils.utils import calc_cnn_out_size


class CLIPLingUNet(nn.Module):
    """ CLIP RN50 with U-Net skip connections """

    def __init__(self, input_shape, output_dim, cfg, device):
        super(CLIPLingUNet, self).__init__()
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
        self.clip_rn50 = self._load_clip()
        self._build_decoder()
    
    def calc_img_enc_size(self):
        test_tensor = torch.zeros(self.input_shape).permute(2, 0, 1)
        test_tensor = test_tensor.to(self.device).unsqueeze(0)
        shape = self.encode_image(test_tensor)[0].shape[1:]
        return shape

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        _clip_rn50 = build_model(model.state_dict()).to(self.device)
        del model
        # Fix encoder weights. Only train decoder
        for param in _clip_rn50.parameters():
            param.requires_grad = False
        return _clip_rn50

    def _build_decoder(self):
        # language
        self.proj_input_dim = 512 if 'word' in self.lang_fusion_type else 1024
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
        self.lang_blocks = []
        in_channels = out_channels * 2
        for i in range(1, 4):
            out_channels = in_channels // 2
            lang_fuser = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // (2 * i)).to(self.device)
            lang_proj = nn.Linear(self.proj_input_dim, out_channels).to(self.device)
            up_conv = Up(in_channels, out_channels // self.up_factor, self.bilinear).to(self.device)
            in_channels = out_channels

            self.lang_blocks.append((lang_fuser, lang_proj, up_conv))

            # Save shape
            h = layers_info[-1]["shape"][-1] * 2
            h = calc_cnn_out_size(h, k=3, p=1)
            h = calc_cnn_out_size(h, k=3, p=1)
            layers_info.append({"name": "conv_lang%d" % i,
                                "shape": (out_channels // self.up_factor, h, h)})


        # Decoder layers w/o language
        self.decoder_blocks = []
        in_channels = 128
        h = self.input_shape[0] // 2
        for i in range(1, 4):
            out_channels = in_channels // 2
            out_c = [out_channels, out_channels, out_channels]
            layer = nn.Sequential(
                ConvBlock(in_channels, out_c, kernel_size=3, stride=1, batchnorm=self.batchnorm),
                IdentityBlock(out_channels, out_c, kernel_size=3, stride=1, batchnorm=self.batchnorm),nn.UpsamplingBilinear2d(scale_factor=2)).to(self.device)
            self.decoder_blocks.append(layer)
            in_channels = out_channels

            # Calc out shape
            layers_info.append({"name": "decoder%d" % i,
                                "shape": (out_channels, h, h)})
            h = layers_info[-1]["shape"][-1] * 2


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

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask

    def forward(self, x, l):
        in_type = x.dtype
        in_shape = x.shape
        x = x[:,:3]  # select RGB
        x, im = self.encode_image(x)
        x = x.to(in_type)

        # encode text
        l_enc, l_emb, l_mask = self.encode_text(l)
        l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=x.dtype)
    
        _info = {"hidden_layers": [x],
                 "l_mask": l_mask,
                 "l_input": l_input,
                 "fusion_type": self.lang_fusion_type}
        # Decoder
        # encode image
        assert x.shape[1] == self.input_dim
        x = self.conv1(x)

        for i, (lang_fuser, lang_proj, up_conv) in enumerate(self.lang_blocks, 2):
            x = lang_fuser(x, l_input, x2_mask=l_mask, x2_proj=lang_proj)
            x = up_conv(x, im[-i])
            _info["hidden_layers"].append(x)

        for layer in self.decoder_blocks:
            x = layer(x)
            _info['hidden_layers'].append(x)

        x = self.conv2(x)
        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        return x, _info