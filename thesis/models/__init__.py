from kornia import depth
from .visual_lang_encoders.rn50_bert_lingunet import RN50BertLingUNet
from .visual_lang_encoders.untrained_rn50_bert_lingunet import UntrainedRN50BertLingUNet

from .visual_lang_encoders.clip_lingunet import CLIPLingUNet

# VAPO enc-dec arch
from .visual_lang_encoders.rn18_clip_lingunet import RN18CLIPLingunet
from .visual_lang_encoders.rn18_bert_lingunet import RN18BertLingunet

lang_img_nets = {
    # Lang Nets
    'untrained_rn50_bert_lingunet': UntrainedRN50BertLingUNet,
    'rn50_bert_lingunet': RN50BertLingUNet,
    'rn50_clip_lingunet': CLIPLingUNet,
    'rn18_clip_lingunet': RN18CLIPLingunet,
    'rn18_bert_lingunet': RN18BertLingunet,
}

# Depth estimatiom models
from .depth.depth_gaussian import DepthEstimationGaussian
from .depth.depth_logistics import DepthEstimationLogistics

deth_est_nets = {
    # Depth Nets
    'gaussian': DepthEstimationGaussian,
    'logistic': DepthEstimationLogistics,
}