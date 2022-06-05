from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from thesis.models.language_encoders.base_lang_encoder import LangEncoder

class SBertLang(LangEncoder):
    def __init__(self, device, fixed=True, pretrained=True) -> None:
        super(SBertLang, self).__init__(device, fixed, pretrained)

    def _load_model(self, cfg):
        self.model = SentenceTransformer(cfg.weights)
        self.text_fc = nn.Linear(768, 1024)

    def encode_text(self, x: List) -> torch.Tensor:
        enc = self.model.encode(x,
                                convert_to_tensor=True,
                                show_progress_bar=False)
        return torch.unsqueeze(enc, 1), None, None