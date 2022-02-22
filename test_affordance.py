"""Main training script."""

import os
import cv2
import hydra
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from thesis.utils.utils import add_img_text, blend_imgs, get_abspath, overlay_mask, load_aff_model
from torch.utils.data import DataLoader
import logging

@hydra.main(config_path="./config", config_name='test_affordance')
def main(cfg):
    # Checkpoint loader
    hydra_run_dir = get_abspath(cfg.checkpoint.train_folder)

    hydra_cfg_path = os.path.join(hydra_run_dir, ".hydra/config.yaml")
    if os.path.exists(hydra_cfg_path):
        run_cfg = OmegaConf.load(hydra_cfg_path)
        run_cfg.aff_detection.dataset.data_dir = cfg.aff_detection.dataset.data_dir
    else:
        print("path does not exist %s" % hydra_cfg_path)
        run_cfg = cfg

    # Load model
    model = load_aff_model(hydra_run_dir,
                           cfg.checkpoint.model_name,
                           run_cfg.aff_detection.model,
                           transforms=run_cfg.aff_detection.dataset.transforms['validation'],
                           hough_voting=cfg.aff_detection.hough_voting)
    model.eval()

    # Dataloaders
    logger = logging.getLogger(__name__)
    val = hydra.utils.instantiate(cfg.aff_detection.dataset, split="validation", log=logger)
    val_loader = DataLoader(val, num_workers=1, batch_size=1, pin_memory=True)
    print("val minibatches {}".format(len(val_loader)))

    cm = plt.get_cmap("jet")
    colors = cm(np.linspace(0, 1, val.n_classes))
    for b_idx, b in enumerate(val_loader):
        # RGB
        inp, labels = b
        frame = inp["orig_frame"][0].detach().cpu().numpy()
        frame = (frame * 255.0).astype("uint8")
        frame = np.transpose(frame, (1, 2, 0))
        frame = cv2.resize(frame, (inp["img"].shape[-2:]))
        if frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        obs = {
            "img": frame,
            "lang_goal": inp["lang_goal"][0]
        }
        out_img = frame.copy()
        for label in range(0, val.n_classes):
            color = colors[label]
            color[-1] = 0.3
            color = tuple((color * 255).astype("int32"))

            # Draw center
            center_px = labels["p0"][0].numpy().squeeze()
            y, x = center_px[0].item(), center_px[1].item()
            out_img = cv2.drawMarker(
                out_img,
                (x, y),
                (0, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        info = None  # labels
        pred = model.predict(obs, info=info)
        pred_img = frame.copy()
        model.viz_preds(obs, pred, waitkey=0)


if __name__ == '__main__':
    main()
