import torch
import torch.nn as nn
from thesis.models.core.clip import build_model, load_clip, tokenize
from thesis.models.language_encoders.base_lang_encoder import LangEncoder


class CLIPLang(LangEncoder):
    def __init__(self, device, fixed=True, pretrained=True) -> None:
        super(CLIPLang, self).__init__(device, fixed, pretrained)

    def _load_model(self):
        model, _ = load_clip("RN50", device=self.device)
        _clip_rn50 = build_model(model.state_dict()).to(self.device)
        del model
        for param in _clip_rn50.parameters():
            param.requires_grad = False
        self.model = _clip_rn50

    def encode_image(self, img):
        with torch.no_grad():
            img_encoding, img_im = self.clip_rn50.visual.prepool_im(img)
        return img_encoding, img_im

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.model.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens==0, tokens, 1)
        return text_feat, text_emb, text_mask