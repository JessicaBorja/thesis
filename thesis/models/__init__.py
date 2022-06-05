# Vision networks
from .visual_lang_encoders.rn50_unet import RN50LingUNet
from .visual_lang_encoders.rn18_unet import RN18Lingunet
from .visual_lang_encoders.rn50_clip_lingunet import CLIPLingUNet

# Language encoders
from .language_encoders.clip_lang_encoder import CLIPLang
from .language_encoders.distilbert_lang_encoder import BERTLang
from .language_encoders.sbert_lang_encoder import SBertLang

lang_encoders = {
    "clip": CLIPLang,
    "bert": BERTLang,
    "sbert": SBertLang
}

vision_encoders = {
    # Lang Nets
    'clip': CLIPLingUNet,
    'rn50': RN50LingUNet,
    'rn18': RN18Lingunet,
}

# Depth estimatiom models
from .depth.depth_gaussian import DepthEstimationGaussian
from .depth.depth_logistics import DepthEstimationLogistics

deth_est_nets = {
    # Depth Nets
    'gaussian': DepthEstimationGaussian,
    'logistic': DepthEstimationLogistics,
}