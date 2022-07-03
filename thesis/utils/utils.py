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
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import re
from scipy.spatial.transform.rotation import Rotation as R
import numpy as np
from os.path import expanduser
import importlib

logger = logging.getLogger(__name__)


def unravel_idx(indices, shape):
    coord = []
    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = np.stack(coord[::-1], axis=-1)
    return coord

def calc_cnn_out_size(in_size, k, p=0, s=1):
    out_size = ((in_size + 2 * p - k) / s) + 1
    return int(out_size)


def np_quat_to_scipy_quat(quat):
    """wxyz to xyzw"""
    return np.array([quat.x, quat.y, quat.z, quat.w])


def pos_orn_to_matrix(pos, orn):
    """
    :param pos: np.array of shape (3,)
    :param orn: np.array of shape (4,) -> quaternion xyzw
                np.quaternion -> quaternion wxyz
                np.array of shape (3,) -> euler angles xyz
    :return: 4x4 homogeneous transformation
    """
    mat = np.eye(4)
    if isinstance(orn, np.quaternion):
        orn = np_quat_to_scipy_quat(orn)
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 4:
        mat[:3, :3] = R.from_quat(orn).as_matrix()
    elif len(orn) == 3:
        mat[:3, :3] = R.from_euler('xyz', orn).as_matrix()
    mat[:3, 3] = pos
    return mat


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


def load_aff_model(hydra_run_dir, model_name, eval=False, **kwargs):
    # Load model
    checkpoint_path = os.path.join(hydra_run_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_path, model_name)
    if(os.path.isfile(checkpoint_path)):
        aff_cfg = os.path.join(hydra_run_dir, ".hydra/config.yaml")
        if os.path.isfile(aff_cfg):
            train_cfg = OmegaConf.load(aff_cfg)
            _model_cfg = train_cfg.aff_detection
        if eval:
            _model_cfg.model_cfg.freeze_encoder.lang=True
            _model_cfg.model_cfg.freeze_encoder.aff=True
            _model_cfg.model_cfg.freeze_encoder.depth=True
        # Get class
        model_class = _model_cfg._target_.split('.')
        model_file = '.'.join(_model_cfg._target_.split('.')[:-1])
        model_file = importlib.import_module(model_file)
        model_class = getattr(model_file, model_class[-1])

        # Instantiate
        model = model_class.load_from_checkpoint(checkpoint_path, **kwargs).cuda()

        # Override default voting layer parameters
        if 'hough_voting' in kwargs and 'hough_voting' in model.model_cfg:
            model.init_voting_layer(kwargs['hough_voting'])
        logger.info("Model successfully loaded: %s" % checkpoint_path)
    else:
        logger.info("No checkpoint file found, loading untrained model: %s" % checkpoint_path)
    if eval:
        model.eval()
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
    '''
        transforms_
    '''
    transforms_lst = []
    transforms_config = transforms_cfg.copy()
    normalize_values, rand_shift = None, None

    for cfg in transforms_config:
        if ("size" in cfg) and img_size is not None:
            cfg.size = [img_size, img_size]
        if("interpolation" in cfg):
            cfg.interpolation = InterpolationMode(cfg.interpolation)
        if("Normalize" in cfg._target_):
            normalize_values = cfg
        if("RandomShift" in cfg._target_):
            rand_shift = hydra.utils.instantiate(cfg)
        else:
            transforms_lst.append(hydra.utils.instantiate(cfg, _convert_="partial"))

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
    path_str = os.path.expanduser(path_str)
    if not os.path.isabs(path_str):
        hydra_cfg = hydra.utils.HydraConfig().cfg
        if hydra_cfg is not None:
            cwd = hydra.utils.get_original_cwd()
        else:
            cwd = os.getcwd()
        path_str = os.path.join(cwd, path_str)
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

def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    pixel_max = np.unravel_index(attn_map.argmax(), attn_map.shape)[:2]
    
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
    return attn_map, pixel_max

def viz_attn(img, attn_map, blur=True):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    attn_map, pixel_max = getAttMap(img, attn_map, blur)
    y, x = pixel_max
    axes[1].plot(x, y,'x', color='black', markersize=12)
    axes[1].imshow(attn_map)
    for ax in axes:
        ax.axis("off")
    plt.show()
    
def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.

def get_aff_model(train_folder, model_name, img_resize=None, eval=True):
    hydra_run_dir = get_abspath(train_folder)
    hydra_cfg_path = os.path.join(hydra_run_dir, ".hydra/config.yaml")
    if os.path.exists(hydra_cfg_path):
        run_cfg = OmegaConf.load(hydra_cfg_path)
    else:
        print("path does not exist %s" % hydra_cfg_path)
        return None, None

    # Load model
    model = load_aff_model(hydra_run_dir,
                           model_name,
                           transforms=run_cfg.aff_detection.dataset.transforms['validation'],
                           eval=eval)
    return model, run_cfg