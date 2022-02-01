import torch
import torch.nn as nn

from thesis.models.core import fusion
import segmentation_models_pytorch as smp
from thesis.models.core.unet_decoder import UnetLangFusionDecoder
from thesis.models.core.clip import build_model, load_clip, tokenize


class RN18CLIPLingunet(nn.Module):
    """Resnet 18 with U-Net skip connections and CLIP language encoder"""

    def __init__(self, input_shape, output_dim, cfg, device):
        super().__init__()
        in_channels = input_shape[-1]
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.cfg = cfg
        self.device = device
        self.lang_fusion_type = self.cfg['lang_fusion_type']

        # Use clip text embeddings
        self.clip_rn50 = self._load_clip()
        self.unet = self._build_model(cfg.decoder_channels, in_channels)

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        _clip_rn50 = build_model(model.state_dict()).to(self.device)
        del model
        # Fix encoder weights. Only train decoder
        for param in _clip_rn50.parameters():
            param.requires_grad = False
        return _clip_rn50

    def _build_model(self, decoder_channels, in_channels):
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

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask


    def forward(self, x, l):
        # in_type = x.dtype
        # in_shape = x.shape
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