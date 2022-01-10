import numpy as np
import hydra
import os
from torchvision import transforms


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