import torch
import torch.nn as nn

from thesis.models.core import fusion
import segmentation_models_pytorch as smp
from thesis.models.core.unet_decoder import UnetLangFusionDecoder
from transformers import DistilBertTokenizer, DistilBertModel


class RN18BertLingunet(nn.Module):
    """Resnet 18 with U-Net skip connections and BERT language encoder"""

    def __init__(self, input_shape, output_dim, cfg, device):
        super().__init__()
        in_channels = input_shape[-1]
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.cfg = cfg
        self.device = device
        self.lang_fusion_type = self.cfg['lang_fusion_type']

        # Use BERT text embeddings
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.lang_embed_dim = 1024
        self.text_fc = nn.Linear(768, self.text_enc_dim)
        self.unet = self._build_model(cfg.decoder_channels, in_channels)

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
        self.decoder = UnetLangFusionDecoder(
            fusion_module = fusion.names[self.lang_fusion_type],
            lang_embed_dim = self.lang_embed_dim,
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
            inputs = self.tokenizer(x, return_tensors='pt')
            text_embeddings = self.text_encoder(**inputs)
            sentence_encodings = text_embeddings.last_hidden_state.mean(1)
        text_feat = self.text_fc(sentence_encodings)
        return text_feat, text_embeddings.last_hidden_state

    def forward(self, x, l):
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
        logits = self.decoder(l_input, *features)
        x = self.conv2d(logits)
        return x