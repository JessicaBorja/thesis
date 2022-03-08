import torch

from thesis.models.rn18_lingunet import RN18Lingunet
from thesis.models.core.clip import build_model, load_clip, tokenize


class RN18CLIPLingunet(RN18Lingunet):
    """Resnet 18 with U-Net skip connections and CLIP language encoder"""

    def __init__(self, input_shape, output_dim, cfg, device):
        super().__init__(input_shape, output_dim, cfg, device)
        # Use clip text embeddings
        self.clip_rn50 = self._load_clip()
        self.lang_embed_dim = self.clip_rn50.state_dict()["text_projection"].shape[1]
        self.unet = self._build_model(cfg.unet_cfg.decoder_channels, self.in_channels)

    def _load_clip(self):
        model, _ = load_clip("RN50", device=self.device)
        _clip_rn50 = build_model(model.state_dict()).to(self.device)
        del model
        # Fix encoder weights. Only train decoder
        for param in _clip_rn50.parameters():
            param.requires_grad = False
        return _clip_rn50

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.clip_rn50.encode_text_with_embeddings(tokens)

        # text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb