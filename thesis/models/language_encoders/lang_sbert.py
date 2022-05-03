from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from thesis.models.language_encoders.lang_enc import LangEncoder

class SBertLang(LangEncoder):
    def __init__(self, cfg, device, fixed=True) -> None:
        super(SBertLang, self).__init__()
        self._load_model(cfg)
        self.device = device
        self.fixed = fixed

    def _load_model(self, cfg):
        self.model = SentenceTransformer(cfg.weights)
        self.text_fc = nn.Linear(768, 1024)

    def forward(self, x: List) -> torch.Tensor:
        enc = self.model.encode(x,
                                convert_to_tensor=True,
                                show_progress_bar=False)
        return torch.unsqueeze(enc, 1), None, None