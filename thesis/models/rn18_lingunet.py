import torch
import torch.nn as nn

from thesis.models.core import fusion
import segmentation_models_pytorch as smp
from thesis.models.core.unet_decoder import UnetLangFusionDecoder
from thesis.models.core.clip import build_model, load_clip, tokenize


class RN18Lingunet(nn.Module):
    """Resnet 18 with U-Net skip connections and [] language encoder"""

    def __init__(self, input_shape, output_dim, cfg, device):
        super().__init__()
        self.in_channels = input_shape[-1]
        self.input_shape = input_shape
        self.n_classes = output_dim
        self.cfg = cfg
        self.device = device
        self.lang_fusion_type = self.cfg['lang_fusion_type']

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
        for param in unet.encoder.parameters():
            param.requires_grad = False
        return unet

    def forward(self, x, l, softmax=True):
        # in_type = x.dtype
        # in_shape = x.shape
        x = x[:,:3]  # select RGB
        features = self.unet.encoder(x)

        # encode text
        l_enc, l_emb = self.encode_text(l)
        l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=x.dtype)
    
        # Decoder
        # encode image
        decoder_feat = self.decoder(l_input, *features)
        aff_out = self.unet.segmentation_head(decoder_feat)

        info = {"decoder": decoder_feat,
               "affordance": aff_out}
        return aff_out, info