import os

import hydra
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform.rotation import Rotation as R
from torchvision import transforms

from affordance.affordance_model import AffordanceModel


def depth_img_from_uint16(depth_img, max_depth=4):
    depth_img[np.isnan(depth_img)] = 0
    return (depth_img.astype("float") / (2 ** 16 - 1)) * max_depth


def get_abs_path(path_str):
    if not os.path.isabs(path_str):
        path_str = os.path.join(hydra.utils.get_original_cwd(), path_str)
        path_str = os.path.abspath(path_str)
    return path_str


def euler_to_quat(euler_angles):
    """xyz euler angles to xyzw quat"""
    return R.from_euler("xyz", euler_angles).as_quat()


def quat_to_euler(quat):
    """xyz euler angles to xyzw quat"""
    return R.from_quat(quat).as_euler("xyz")