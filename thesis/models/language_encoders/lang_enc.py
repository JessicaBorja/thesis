import torch
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import torch.nn as nn


class LangEncoder(nn.Module):
    def __init__(self, device, fixed=True) -> None:
        super(LangEncoder, self).__init__()
        self._load_model()
        self.device = device
        self.fixed = fixed

    def encode(self, x):
        ''' Get sentence encodings for a given annotation'''
        text_enc, text_embeddings, text_mask = self.forward(x)
        return text_enc
    
    def _load_model(self):
        raise NotImplementedError()

    def forward(self, x):
        '''
            Returns:
                - text_encodings
                - text_embeddings
                - text_mask
        '''
        raise NotImplementedError()