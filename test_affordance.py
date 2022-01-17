"""Main training script."""

import os
import cv2
import hydra
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from omegaconf import OmegaConf
from thesis.utils.utils import get_hydra_launch_dir, overlay_mask
from thesis.utils.utils import load_aff_model
from thesis.datasets.calvin_data import CalvinDataLang, DataLoader


@hydra.main(config_path="./config", config_name='test_affordance')
def main(cfg):
    # Checkpoint loader
    hydra_run_dir = get_hydra_launch_dir(cfg.checkpoint.path)

    hydra_cfg_path = os.path.join(hydra_run_dir, ".hydra/config.yaml")
    if os.path.exists(hydra_cfg_path):
        run_cfg = OmegaConf.load(hydra_cfg_path)
        run_cfg.dataset.data_dir = cfg.dataset.data_dir
    else:
        print("path does not exist %s" % hydra_cfg_path)
        run_cfg = cfg

    # Load model
    model = load_aff_model(hydra_run_dir,
                           cfg.checkpoint.model_name,
                           run_cfg.aff_detection)
    model.eval()

    # Dataloaders
    val = CalvinDataLang(split="validation", log=None, **run_cfg.dataset)
    val_loader = DataLoader(val, num_workers=1, batch_size=1, pin_memory=True)
    print("val minibatches {}".format(len(val_loader)))

    cm = plt.get_cmap("jet")
    colors = cm(np.linspace(0, 1, val.n_classes))
    for b_idx, b in enumerate(val_loader):
        # RGB
        inp, labels = b
        frame = inp["img"][0].detach().cpu().numpy()
        frame = (frame * 255).astype("uint8")
        frame = np.transpose(frame, (1, 2, 0))
        if frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        obs = {"img": frame.copy(),
               "lang_goal": inp["lang_goal"][0]}
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
        pred = model.predict(obs)
        pred_img = frame.copy()
        pred_img = overlay_mask(pred["softmax"], pred_img, (0, 0, 255))
        pixel = pred["pixel"]
        pred_img = cv2.drawMarker(
                pred_img,
                (pixel[0], pixel[1]),
                (0, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        pred_img = cv2.resize(pred_img, (300, 300), interpolation=cv2.INTER_CUBIC)
        out_img = cv2.resize(out_img, (300, 300), interpolation=cv2.INTER_CUBIC)

        out_img = np.hstack([out_img, pred_img])

        # Prints the text.
        font_scale = 0.6
        thickness = 2
        color = (0, 0, 0)
        x1, y1 = 10, 20
        text_label = obs["lang_goal"]
        (w, h), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        out_img = cv2.rectangle(out_img, (x1, y1 - 20), (x1 + w, y1 + h), color, -1)
        out_img = cv2.putText(
            out_img,
            text_label,
            org=(x1, y1),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=thickness,
        )

        cv2.imshow("img", out_img[:, :, ::-1])
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
