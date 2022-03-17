import torch
import torch.nn as nn

from .rn18_lingunet import RN18Lingunet
from transformers import DistilBertTokenizer, DistilBertModel


class RN18BertLingunet(RN18Lingunet):
    """Resnet 18 with U-Net skip connections and BERT language encoder"""

    def __init__(self, input_shape, output_dim, cfg, device):
        super().__init__(input_shape, output_dim, cfg, device)
        # Use BERT text embeddings
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
        self.lang_embed_dim = 1024
        self.text_fc = nn.Linear(768, self.lang_embed_dim)
        self.unet = self._build_model(cfg.unet_cfg.decoder_channels, self.in_channels)

    def encode_text(self, x):
        with torch.no_grad():
            inputs = self.tokenizer(x,
                                    return_tensors='pt',
                                    padding='max_length',
                                    truncation=True).to(self.device)
            text_embeddings = self.text_encoder(**inputs)
            sentence_encodings = text_embeddings.last_hidden_state.mean(1)
        text_feat = self.text_fc(sentence_encodings)
        return text_feat, text_embeddings.last_hidden_state