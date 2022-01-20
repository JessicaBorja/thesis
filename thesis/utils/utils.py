import numpy as np
import hydra
import os
from torchvision import transforms
import torch
from torch.autograd import Variable
from PIL import Image
import logging
import cv2
from torchvision.transforms import InterpolationMode

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


def blend_imgs(background, foreground, alpha=0.5):
    """
    Blend two images of the same shape with an alpha value
    img1: np.array(uint8)
        - shape: (H, W)
        - range: 0 - 255
    img1: np.array(uint8)
        - shape: (H, W, 3)
        - range: 0 - 255
    alpha(float): (0, 1) value
    """
    foreground = foreground.astype(float)
    background = background.astype(float)

    alpha = np.ones_like(foreground, dtype=float) * alpha # alpha.astype(float)/255
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    outImage = cv2.add(foreground, background)/255
    # outImage = ((outImage + 1)/2 * 255).astype('uint8')
    return outImage


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
    normalize_values, rand_shift = None, None

    for cfg in transforms_config:
        if ("size" in cfg) and img_size is not None:
            cfg.size = img_size
        if("interpolation" in cfg):
            cfg.interpolation = InterpolationMode(cfg.interpolation)
        if("Normalize" in cfg._target_):
            normalize_values = cfg
        if("RandomShift" in cfg._target_):
            rand_shift = hydra.utils.instantiate(cfg)
        else:
            transforms_lst.append(hydra.utils.instantiate(cfg))

    return  {
        "transforms": transforms.Compose(transforms_lst),
        "norm_values": normalize_values,
        "rand_shift": rand_shift
    }


def get_hydra_launch_dir(path_str):
    if not os.path.isabs(path_str):
        path_str = os.path.join(hydra.utils.get_original_cwd(), path_str)
        path_str = os.path.abspath(path_str)
    return path_str


def pixel_after_pad(pixel, pad):
    l, r, t, b = pad
    pad_val = np.array((l, t))
    new_pixel = pixel + pad_val
    return new_pixel


def resize_pixel(pixel, old_shape, new_shape):
    assert len(old_shape) == len(new_shape)
    c = np.array(pixel) * new_shape // old_shape
    return c