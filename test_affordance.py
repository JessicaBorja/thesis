"""Main training script."""

import os
import cv2
import hydra
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from omegaconf import OmegaConf
from thesis.utils.utils import blend_imgs, get_hydra_launch_dir, overlay_mask, load_aff_model
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
                           run_cfg.aff_detection.model,
                           transforms=run_cfg.aff_detection.transforms['validation'])
    model.eval()

    # Dataloaders
    val = CalvinDataLang(split="validation", log=None, **run_cfg.dataset)
    val_loader = DataLoader(val, num_workers=1, batch_size=1, shuffle=True, pin_memory=True)
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
            "lang_goal": inp["lang_goal"]
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

        cm = plt.get_cmap('viridis')
        heatmap = cm(pred["softmax"])[:, :, [0,1,2]] * 255
        heatmap = blend_imgs(frame.copy(), heatmap, alpha=0.7)

        pixel = pred["pixel"]
        # print(pred["error"], pred["pixel"], (x, y))
        pred_img = cv2.drawMarker(
                pred_img,
                (pixel[0], pixel[1]),
                (0, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

        new_size = (400, 400)
        heatmap = (heatmap * 255).astype('uint8')
        pred_img = cv2.resize(pred_img, new_size, interpolation=cv2.INTER_CUBIC)
        out_img = cv2.resize(out_img, new_size, interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.resize(heatmap, new_size, interpolation=cv2.INTER_CUBIC)
        # out_img = out_img.astype(float) / 255
        # pred_img = pred_img.astype(float) / 255
        out_img = np.concatenate([out_img, pred_img, heatmap], axis=1)

        # Prints the text.
        font_scale = 0.6
        thickness = 2
        color = (0, 0, 0)
        x1, y1 = 10, 20
        text_label = inp['lang_goal'][0]
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

        out_img = out_img[:, :, ::-1]

        # file_dir = "./imgs"
        # os.makedirs(file_dir, exist_ok=True)
        # filename = os.path.join(file_dir, "img_%04d.png" % b_idx)
        # cv2.imwrite(filename, out_img)

        cv2.imshow("img", out_img)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
