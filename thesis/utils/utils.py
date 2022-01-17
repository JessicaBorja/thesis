import numpy as np
import hydra
import os
from torchvision import transforms
import torch
from torch.autograd import Variable
from PIL import Image
import logging
logger = logging.getLogger(__name__)


def load_aff_model(hydra_run_dir, model_name, model_cfg):
    # Load model
    checkpoint_path = os.path.join(hydra_run_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_path, model_name)
    if(os.path.isfile(checkpoint_path)):
        model = hydra.utils.instantiate(model_cfg)
        model = model.load_from_checkpoint(checkpoint_path).cuda()
        logger.info("Model successfully loaded: %s" % checkpoint_path)
    else:
        model = hydra.utils.instantiate(model_cfg).cuda()
        logger.info("No checkpoint file found, loading untrained model: %s" % checkpoint_path)
    return model


def overlay_mask(mask, img, color):
    """
    mask: np.array
        - shape: (H, W)
        - range: 0 - 255.0
        - uint8
    img: np.array
        - shape: (H, W, 3)
        - range: 0 - 255
        - uint8
    color: tuple
        - tuple size 3 RGB
        - range: 0 - 255
    """
    result = Image.fromarray(np.uint8(img))
    pil_mask = Image.fromarray(np.uint8(mask))
    color = Image.new("RGB", result.size, color)
    result.paste(color, (0, 0), pil_mask)
    result = np.array(result)
    return result


def tt(x, device):
    if isinstance(x, dict):
        dict_of_list = {}
        for key, val in x.items():
            dict_of_list[key] = Variable(torch.from_numpy(val).float().to(device),
                                         requires_grad=False)
        return dict_of_list
    else:
        return Variable(torch.from_numpy(x).float().to(device),
                        requires_grad=False)


def get_transforms(transforms_cfg, img_size=None):
    transforms_lst = []
    transforms_config = transforms_cfg.copy()
    for cfg in transforms_config:
        if (cfg._target_ == "torchvision.transforms.Resize" or "RandomCrop" in cfg._target_) and img_size is not None:
            cfg.size = img_size
        if "vapo.affordance_model.utils.transforms" in cfg._target_:
            cfg._target_ = cfg._target_.replace(
                "vapo.affordance_model.utils.transforms",
                "affordance.dataloader.transforms",
            )
        transforms_lst.append(hydra.utils.instantiate(cfg))

    return transforms.Compose(transforms_lst)


def get_hydra_launch_dir(path_str):
    if not os.path.isabs(path_str):
        path_str = os.path.join(hydra.utils.get_original_cwd(), path_str)
        path_str = os.path.abspath(path_str)
    return path_str


def resize_pixel(pixel, old_shape, new_shape):
    assert len(old_shape) == len(new_shape)
    c = np.array(pixel) * new_shape // old_shape
    return c