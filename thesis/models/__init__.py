from .resnet import ResNet43_8s
from .clip_wo_skip import CLIPWithoutSkipConnections

from .rn50_bert_unet import RN50BertUNet
from .rn50_bert_lingunet import RN50BertLingUNet
from .rn50_bert_lingunet_lat import RN50BertLingUNetLat
from .untrained_rn50_bert_lingunet import UntrainedRN50BertLingUNet

from .clip_unet import CLIPUNet
from .clip_lingunet import CLIPLingUNet

from .resnet_lang import ResNet43_8s_lang

from .resnet_lat import ResNet45_10s
from .clip_unet_lat import CLIPUNetLat
from .clip_lingunet_lat import CLIPLingUNetLat
from .clip_film_lingunet_lat import CLIPFilmLingUNet

from .unet_lang import UnetLang


names = {
    # resnet
    'plain_resnet': ResNet43_8s,
    'plain_resnet_lang': ResNet43_8s_lang,

    # without skip-connections
    'clip_woskip': CLIPWithoutSkipConnections,

    # unet
    'clip_unet': CLIPUNet,
    'rn50_bert_unet': RN50BertUNet,

    # lingunet
    'clip_lingunet': CLIPLingUNet,
    'rn50_bert_lingunet': RN50BertLingUNet,
    'untrained_rn50_bert_lingunet': UntrainedRN50BertLingUNet,
    'unet_lang': UnetLang,

    # lateral connections
    'plain_resnet_lat': ResNet45_10s,
    'clip_unet_lat': CLIPUNetLat,
    'clip_lingunet_lat': CLIPLingUNetLat,
    'clip_film_lingunet_lat': CLIPFilmLingUNet,
    'rn50_bert_lingunet_lat': RN50BertLingUNetLat,
}
