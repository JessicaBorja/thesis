import torch
import torch.nn as nn

from thesis.models.core import fusion
import segmentation_models_pytorch as smp
from thesis.models.core.unet_decoder import UnetLangFusionDecoder
from thesis.models.visual_lang_encoders.base_lingunet import BaseLingunet


class RN18Lingunet(BaseLingunet):
    """Resnet 18 with U-Net skip connections and [] language encoder"""

    def __init__(self, input_shape, output_dim, cfg, device):
        super(RN18Lingunet, self).__init__(input_shape, output_dim, cfg, device)
        self.in_channels = input_shape[-1]
        self.n_classes = output_dim
        self.lang_embed_dim = 1024
        self.text_fc = nn.Linear(768, self.lang_embed_dim)
        self.freeze_backbone = cfg.freeze_encoder.aff
        self.unet = self._build_model(cfg.unet_cfg.decoder_channels, self.in_channels)

    def _build_model(self, decoder_channels, in_channels):
        # encoder_depth Should be equal to number of layers in decoder
        unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=self.n_classes,
            encoder_depth=len(decoder_channels),
            decoder_channels=tuple(decoder_channels),
            activation=None,
        )

        self.decoder = UnetLangFusionDecoder(
            fusion_module = fusion.names[self.lang_fusion_type],
            lang_embed_dim = self.lang_embed_dim,
            encoder_channels=unet.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels))

        self.decoder_channels = decoder_channels
        # Fix encoder weights. Only train decoder
        if self.freeze_backbone:
            for param in unet.encoder.parameters():
                param.requires_grad = False
            for param in unet.encoder.layer4.parameters():
                param.requires_grad = True
        return unet

    def forward(self, x, text_enc, softmax=True):
        # in_type = x.dtype
        # in_shape = x.shape
        x = x[:,:3]  # select RGB
        features = self.unet.encoder(x)

        # Language encoding
        l_enc, l_emb, l_mask  = text_enc
        l_input = l_enc.to(dtype=x.dtype)
    
        # Decoder
        # encode image
        decoder_feat = self.decoder(l_input, *features)
        aff_out = self.unet.segmentation_head(decoder_feat)

        info = {"decoder_out": [decoder_feat],
                "affordance": aff_out,
                "text_enc": l_input}
        return aff_out, info