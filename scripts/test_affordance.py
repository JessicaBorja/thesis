"""Main training script."""

import os
import cv2
import hydra
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from thesis.utils.utils import get_abspath, load_aff_model
from torch.utils.data import DataLoader
import logging


@hydra.main(config_path="../config", config_name='test_affordance')
def main(cfg):
    # Checkpoint loader
    hydra_run_dir = get_abspath(cfg.checkpoint.train_folder)

    hydra_cfg_path = os.path.join(hydra_run_dir, ".hydra/config.yaml")
    if os.path.exists(hydra_cfg_path):
        run_cfg = OmegaConf.load(hydra_cfg_path)
        run_cfg.aff_detection.dataset.data_dir = cfg.aff_detection.dataset.data_dir
    else:
        print("path does not exist %s" % hydra_cfg_path)
        return

    # Load model
    model = load_aff_model(hydra_run_dir,
                           cfg.checkpoint.model_name,
                           run_cfg.aff_detection,
                           transforms=run_cfg.aff_detection.dataset.transforms['validation'],
                           hough_voting=cfg.aff_detection.hough_voting)
    model.eval()

    # Dataloaders
    logger = logging.getLogger(__name__)
    val = hydra.utils.instantiate(run_cfg.aff_detection.dataset, split="validation", log=logger)
    val_loader = DataLoader(val, num_workers=1, batch_size=1, pin_memory=True)
    print("val minibatches {}".format(len(val_loader)))

    cm = plt.get_cmap("jet")
    n_classes = 2
    colors = cm(np.linspace(0, 1, 2))
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
        for label in range(0, n_classes):
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
        out_shape = (400, 400)
        pred_img = model.get_preds_viz(obs, pred, gt_depth=labels["depth"],out_shape=out_shape)
        label_img = cv2.resize(out_img, out_shape) / 255

        out_img = np.concatenate([pred_img, label_img], axis=1)
        cv2.imshow("img", out_img[:, :, ::-1])
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
