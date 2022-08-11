from r3m import load_r3m
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from transformers import DistilBertTokenizer, DistilBertModel
from thesis.models.core.resnet import IdentityBlock, ConvBlock

from thesis.models.core.unet import Up
from thesis.models.core.unet_decoder import UnetLangFusionDecoder
from thesis.models.core import fusion
from torchvision import transforms
from thesis.models.visual_lang_encoders.base_lingunet import BaseLingunet


class R3M(BaseLingunet):
    """ R3M RN 18 & SBert with U-Net skip connections """

    def __init__(self, input_shape, output_dim, cfg, device):
        super(R3M, self).__init__(input_shape, output_dim, cfg)
        self.output_dim = output_dim
        self.input_dim = 512
        self.lang_embed_dim = 1024
        self.batchnorm = self.cfg['batchnorm']
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1
        _encoder_name = "resnet18"
        self.freeze_backbone = cfg.freeze_encoder.aff
        self.vision_normlayer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self._load_vision(_encoder_name, device)
        self._build_decoder(cfg.unet_cfg.decoder_channels)

    def _load_vision(self, resnet_model, device):
        self.r3m = load_r3m(resnet_model, device).module
        modules = list(list(self.r3m.children())[3].children())

        self.stem = nn.Sequential(*modules[:4])
        self.layer1 = modules[4]
        self.layer2 = modules[5]
        self.layer3 = modules[6]
        self.layer4 = modules[7]

        # Fix encoder weights. Only train decoder
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in layer.parameters():
                param.requires_grad = False
        if not self.freeze_backbone:
            for param in self.layer4.parameters():
                param.requires_grad = True

    def calc_img_enc_size(self):
        test_tensor = torch.zeros(self.input_shape).permute(2, 0, 1)
        test_tensor = test_tensor.unsqueeze(0).to(self.r3m.device)
        shape = self.r3m_resnet18(test_tensor)[0].shape[1:]
        return shape

    def _build_decoder(self, decoder_channels):
        # decoder_channels = (256, 128, 64, 64, 32)
        decoder_channels = (512, 256, 128, 64, 32)        
        self.decoder = UnetLangFusionDecoder(
            fusion_module=fusion.names[self.lang_fusion_type],
            lang_embed_dim=self.lang_embed_dim,
            encoder_channels=(3, 64, 64, 128, 256, 512),
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels))

        kernel_size = 3
        n_classes = 1
        self.segmentation_head = nn.Conv2d(decoder_channels[-1],
                                           n_classes,
                                           kernel_size=kernel_size,
                                           padding=kernel_size // 2)
        # self.conv1_dec = nn.Sequential(
        #     nn.Conv2d(self.input_dim, 512, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ReLU(True)
        # )
        # self.up1 = Up(768, 512 // self.up_factor, self.bilinear)
        # self.up2 = Up(384, 256 // self.up_factor, self.bilinear)
        # self.up3 = Up(192, 128 // self.up_factor, self.bilinear)
        #
        # self.layer1_dec =  nn.Sequential(
        #     ConvBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        #     IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        # )
        #
        # self.layer2_dec = nn.Sequential(
        #     ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        #     IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        # )
        #
        # self.layer3_dec = nn.Sequential(
        #     ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        #     IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        # )
        #
        # self.conv2_dec = nn.Sequential(
        #     nn.Conv2d(16, self.output_dim, kernel_size=1)
        # )

    def r3m_resnet18(self, x):
        im = []
        # preprocess = nn.Sequential(
        #     self.vision_normlayer,
        # )
        # ## Input must be [0, 255], [3,244,244]
        # x = x.float() / 255.0
        # x = preprocess(x)
        for layer in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
            im.append(x)
        return x, im

    def encode_image(self, img):
        with torch.no_grad():
            # print("forward shape: ", img.shape)
            # print("max: ", torch.max(img))
            # print("min: ", torch.min(img))
            img_encoding, img_im = self.r3m_resnet18(img)
            # print("forward 2: ", img_encoding.shape)
        return img_encoding, img_im

    def forward(self, x, text_enc):
        in_type = x.dtype
        in_shape = x.shape
        input = x[:,:3]  # select RGB
        #input 32, 3, 224, 224
        x, im = self.encode_image(input) # 32, 512, 7, 7
        x = x.to(in_type)

        # encode language
        l_enc, l_emb, l_mask = text_enc
        l_input = l_enc.to(dtype=x.dtype)  # [32, 1024]

        encoder_feat = [input, *im]
        decoder_feat = self.decoder(l_input, *encoder_feat)
        aff_out = self.segmentation_head(decoder_feat)

        info = {"decoder_out": [decoder_feat],
                "hidden_layers": encoder_feat,
                "affordance": aff_out,
                "text_enc": l_input}
        # # encode image
        # assert x.shape[1] == self.input_dim
        # x = self.conv1_dec(x)  # [32, 512, 7, 7]
        # x = self.lang_fuser1(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj1)
        # x = self.up1(x, im[-2])  # 32, 256, 14, 14
        #
        # x = self.lang_fuser2(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj2)
        # x = self.up2(x, im[-3])  # 32, 128, 28, 28
        #
        # x = self.lang_fuser3(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj3)
        # x = self.up3(x, im[-4])  # 32, 64, 56, 56
        #
        # for layer in [self.layer1_dec, self.layer2_dec, self.layer3_dec, self.conv2_dec]:
        #     print("layer")
        #     x = layer(x)
        #     print(x.shape)
        # # shape: torch.Size([32, 1, 448, 448])
        # x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        # print("finakl: ", x.shape)
        return aff_out, info