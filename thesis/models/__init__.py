from .resnet import ResNet43_8s

from .rn50_bert_unet import RN50BertUNet
from .rn50_bert_lingunet import RN50BertLingUNet
from .untrained_rn50_bert_lingunet import UntrainedRN50BertLingUNet

from .clip_lingunet import CLIPLingUNet

# VAPO enc-dec arch
from .rn18_clip_lingunet import RN18CLIPLingunet
from .rn18_bert_lingunet import RN18BertLingunet


names = {
    # resnet
    'plain_resnet': ResNet43_8s,

    # unet
    'rn50_bert_unet': RN50BertUNet,

    # lingunet
    'clip_lingunet': CLIPLingUNet,
    'rn50_bert_lingunet': RN50BertLingUNet,
    'untrained_rn50_bert_lingunet': UntrainedRN50BertLingUNet,
    'rn18_clip_lingunet': RN18CLIPLingunet,
    'rn18_bert_lingunet': RN18BertLingunet,
}
