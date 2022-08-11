# Vision networks
from .visual_lang_encoders.rn50_unet import RN50LingUNet
from .visual_lang_encoders.rn_lingunet import RNLingunet
from .visual_lang_encoders.rn50_clip_lingunet import CLIPLingUNet
from .visual_lang_encoders.r3m_rn18 import R3M

# Language encoders
from .language_encoders.clip_lang_encoder import CLIPLang
from .language_encoders.bert_lang_encoder import BERTLang
from .language_encoders.distilbert_lang_encoder import DistilBERTLang
from .language_encoders.sbert_lang_encoder import SBertLang

lang_encoders = {
    "clip": CLIPLang,
    "bert": BERTLang,
    "distilbert": DistilBERTLang,
    "sbert": SBertLang
}

vision_encoders = {
    # Lang Nets
    'clip': CLIPLingUNet,
    'rn': RNLingunet, # RN50LingUNet,
    'rn18': RNLingunet,
    'r3m_rn18': R3M,
}

# Depth estimatiom models
from .depth.depth_gaussian import DepthEstimationGaussian
from .depth.depth_logistics import DepthEstimationLogistics

deth_est_nets = {
    # Depth Nets
    'gaussian': DepthEstimationGaussian,
    'logistic': DepthEstimationLogistics,
}