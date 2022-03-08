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
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

def add_img_text(img, text_label):
    font_scale = 0.6
    thickness = 2
    color = (0, 0, 0)
    x1, y1 = 10, 20
    (w, h), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    out_img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1 + h), color, -1)
    out_img = cv2.putText(
        out_img,
        text_label,
        org=(x1, y1),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=thickness,
    )
    return out_img


def load_aff_model(hydra_run_dir, model_name, model_cfg, **kwargs):
    # Load model
    checkpoint_path = os.path.join(hydra_run_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_path, model_name)
    if(os.path.isfile(checkpoint_path)):
        aff_cfg = os.path.join(hydra_run_dir, ".hydra/config.yaml")
        if os.path.isfile(aff_cfg):
            train_cfg = OmegaConf.load(aff_cfg)
            model_cfg = train_cfg.aff_detection.model
        model = hydra.utils.instantiate(model_cfg, **kwargs)
        model = model.load_from_checkpoint(checkpoint_path, **kwargs).cuda()
        # Override default voting layer parameters
        if 'hough_voting' in kwargs and 'hough_voting' in model.cfg:
            model.init_voting_layer(kwargs['hough_voting'])
        logger.info("Model successfully loaded: %s" % checkpoint_path)
    else:
        model = hydra.utils.instantiate(model_cfg).cuda()
        logger.info("No checkpoint file found, loading untrained model: %s" % checkpoint_path)
    return model


def blend_imgs(background, foreground, alpha=0.5):
    """
    Blend two images of the same shape with an alpha value
    background: np.array(uint8)
        - shape: (H, W)
        - range: 0 - 255
    foreground: np.array(uint8)
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


def torch_to_numpy(x):
    return x.detach().cpu().numpy()


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


def overlay_flow(flow, img, mask):
    """
    Args:
        flow: numpy array, shape = (W, H, 3), between 0 - 255
        img: numpy array, shape = (W, H, 3), between 0 - 255
        mask: numpy array, shape = (W, H), between 0 - 255
    return:
        res: Overlay of mask over image, shape = (W, H, 3), 0-255
    """
    result = Image.fromarray(np.uint8(img.squeeze()))
    pil_mask = Image.fromarray(np.uint8(mask.squeeze()))
    flow = Image.fromarray(np.uint8(flow))
    result.paste(flow, (0, 0), pil_mask)
    result = np.array(result)
    return result


def get_abspath(path_str):
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