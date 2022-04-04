from .visual_lang_encoders.rn50_bert_lingunet import RN50BertLingUNet
from .visual_lang_encoders.untrained_rn50_bert_lingunet import UntrainedRN50BertLingUNet

from .visual_lang_encoders.clip_lingunet import CLIPLingUNet

# VAPO enc-dec arch
from .visual_lang_encoders.rn18_clip_lingunet import RN18CLIPLingunet
from .visual_lang_encoders.rn18_bert_lingunet import RN18BertLingunet


names = {
    'rn50_bert_lingunet': RN50BertLingUNet,
    'untrained_rn50_bert_lingunet': UntrainedRN50BertLingUNet,
    'clip_lingunet': CLIPLingUNet,
    'rn18_clip_lingunet': RN18CLIPLingunet,
    'rn18_bert_lingunet': RN18BertLingunet,
}
