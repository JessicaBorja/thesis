import torch
import torch.nn as nn
from thesis.models.core.clip import build_model, load_clip, tokenize
from thesis.models.language_encoders.lang_enc import LangEncoder


class CLIPLang(LangEncoder):
    def __init__(self, device, fixed=True) -> None:
        super(CLIPLang, self).__init__()
        self.fixed = fixed
        self.device = device
        self._load_model()

    def _load_model(self):
        model, _ = load_clip("RN50", device=self.device)
        _clip_rn50 = build_model(model.state_dict()).to(self.device)
        del model
        for param in _clip_rn50.parameters():
            param.requires_grad = False
        self.model = _clip_rn50

    def forward(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat, text_emb = self.model.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens==0, tokens, 1)
        return text_feat, text_emb, text_mask