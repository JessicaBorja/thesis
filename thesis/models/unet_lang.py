import torch.nn as nn
import torch.nn.functional as F

from thesis.models.core import fusion
from thesis.models.clip_lingunet_lat import CLIPLingUNetLat
import segmentation_models_pytorch as smp
from thesis.models.core.unet_decoder import UnetLangFusionDecoder

class UnetLang(CLIPLingUNetLat):
    """ CLIP RN50 with U-Net skip connections and lateral connections without language """

    def __init__(self, input_shape, output_dim, cfg, device):
        super().__init__(input_shape, output_dim, cfg, device)
        in_channels = input_shape[-1]
        self.unet = self._build_model(cfg.decoder_channels, in_channels)

    def _build_model(self, decoder_channels, in_channels):
        # if decoder_channels is None:
        decoder_channels = [256, 128, 64, 32, 16]
        # encoder_depth Should be equal to number of layers in decoder
        unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=1,
            encoder_depth=len(decoder_channels),
            decoder_channels=tuple(decoder_channels),
            activation=None,
        )

        lang_embed_dim = self.clip_rn50.state_dict()["text_projection"].shape[1]
        self.decoder = UnetLangFusionDecoder(
            fusion_module = fusion.names[self.lang_fusion_type],
            lang_embed_dim = lang_embed_dim,
            encoder_channels=unet.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels))
        self.conv2d = nn.Conv2d(decoder_channels[-1], self.output_dim, kernel_size=1)

        self.decoder_channels = decoder_channels
        # Fix encoder weights. Only train decoder
        for param in unet.encoder.parameters():
            param.requires_grad = False
        return unet

    def encode_image(self, x):
        """Run until prepool and save intermediate features"""
        im = []
        x = x.type(self.conv1.weight.dtype)
        for layer in self.unet.encoder:
            x = layer(x)
            im.append(x)
        # def stem(x):
        #     for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
        #         x = self.relu(bn(conv(x)))
        #         im.append(x)
        #     x = self.avgpool(x)
        #     im.append(x)
        #     return x

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
            im.append(x)

        return x, im

    def forward(self, x, l):
        in_type = x.dtype
        in_shape = x.shape
        x = x[:,:3]  # select RGB
        features = self.unet.encoder(x)

        # encode text
        l_enc, l_emb, _ = self.encode_text(l)
        l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=x.dtype)
    
        # Decoder
        # encode image
        logits = self.decoder(l_input, *features)
        x = self.conv2d(logits)
        return x