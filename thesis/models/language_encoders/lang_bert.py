import torch
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import torch.nn as nn


class BERTLang(nn.Module):
    def __init__(self, device, fixed=True) -> None:
        super(BERTLang, self).__init__()
        self._load_model()
        self.device = device
        self.fixed = fixed

    def _load_model(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_fc = nn.Linear(768, 1024)

    def forward(self, x):
        with torch.no_grad():
            inputs = self.tokenizer(x, return_tensors='pt')
            input_ids, attention_mask = inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)
            text_embeddings = self.text_encoder(input_ids, attention_mask)
            text_encodings = text_embeddings.last_hidden_state.mean(1)
        text_feat = self.text_fc(text_encodings)
        text_mask = torch.ones_like(input_ids) # [1, max_token_len]
        return text_feat, text_embeddings.last_hidden_state, text_mask