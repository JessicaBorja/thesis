import os
import json
import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import logging
from thesis.utils.utils import resize_pixel, get_hydra_launch_dir, get_transforms


class CalvinDataLang(Dataset):
    def __init__(
        self,
        img_resize,
        data_dir,
        transforms_cfg,
        n_train_ep=-1,
        split="training",
        cam="static",
        log=None,
        radius=None,
    ):
        super(CalvinDataLang, self).__init__()
        self.cam = cam
        self.split = split
        self.log = log
        self.data_dir = get_hydra_launch_dir(data_dir)
        _data_info = self.read_json(os.path.join(self.data_dir, "episodes_split.json"))
        self.data = self._get_split_data(_data_info, split, cam, n_train_ep)
        self.img_resize = img_resize
        self.transforms = get_transforms(transforms_cfg[split], img_resize[cam])
        self.out_shape = self.get_channels(img_resize[cam])
        # Excludes background
        self.n_classes = _data_info["n_classes"] if cam == "static" else 1
        self.resize = (self.img_resize[self.cam], self.img_resize[self.cam])
        self.cmd_log = logging.getLogger(__name__)
        self.cmd_log.info("Dataloader using shape: %s" % str(self.resize))

    def read_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data

    def get_channels(self, in_size):
        test_tensor = torch.zeros((3, in_size, in_size))
        test_tensor = self.transforms(test_tensor)
        return test_tensor.shape  # C, H, W

    def _get_split_data(self, data, split, cam, n_train_ep):
        split_data = []
        split_episodes = list(data[split].keys())

        # Select amount of data to train on
        if n_train_ep > 0 and split == "training":
            assert len(split_episodes) >= n_train_ep, "n_train_ep must <= %d" % len(split_episodes)
            split_episodes = np.random.choice(split_episodes, n_train_ep, replace=False)

        print("%s episodes: %s" % (split, str(split_episodes)))
        for ep in split_episodes:
            data[split][ep].sort()
            for file in data[split][ep]:
                if cam in file or cam == "all":
                    split_data.append("%s/%s" % (ep, file))
        print("%s images: %d" % (split, len(split_data)))
        return split_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # directions: optical flow image in middlebury color

        head, filename = os.path.split(self.data[idx].replace("\\", "/"))
        episode, cam_folder = os.path.split(head)
        data = np.load(self.data_dir + "/%s/data/%s/%s.npz" % (episode, cam_folder, filename))

        # Images are stored in BGR
        old_shape = data["frame"].shape[:2]
        frame = data["frame"]
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # C, W, H
        frame = self.transforms(frame)

        # Aff mask
        # data["centers"] = (label, x, y)
        center = resize_pixel(data["centers"][0, 1:], old_shape, self.resize)
        assert (center < self.resize).all(), "center is out of range, old_shape %s, resize %s, old_center %s, new_center %s" % (str(old_shape), str(self.resize), str(data["centers"][0, 1:]), str(center))
        # mask = np.zeros(self.resize)
        # mask[center[0], center[1]] = 1

        # Select a language annotation
        annotations = [i.item() for i in data["lang_ann"]]
        assert len(annotations) > 0, "no language annotation in %s" % self.data[idx]
        lang_ann = np.random.choice(annotations).item()

        task = data["task"].tolist()
        inp = {"img": frame,
               "lang_goal": lang_ann}

        # CE Loss requires mask in form (B, H, W)
        labels = {"task": task[0],
                  "p0": center,
                  "tetha0": []}
        return inp, labels


@hydra.main(config_path="../../config", config_name="train_affordance")
def main(cfg):
    val = CalvinDataLang(split="validation", log=None, **cfg.dataset)
    val_loader = DataLoader(val, num_workers=1, batch_size=2, pin_memory=True)
    print("val minibatches {}".format(len(val_loader)))

    cm = plt.get_cmap("jet")
    colors = cm(np.linspace(0, 1, val.n_classes))
    for b_idx, b in enumerate(val_loader):
        # RGB
        inp, labels = b
        frame = inp["img"][0].detach().cpu().numpy()
        frame = ((frame + 1) * 255 / 2).astype("uint8")
        frame = np.transpose(frame, (1, 2, 0))
        if frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

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

        out_img = cv2.resize(out_img, (500, 500), interpolation=cv2.INTER_CUBIC)

        # Prints the text.
        font_scale = 0.6
        thickness = 2
        color = (0, 0, 0)
        x1, y1 = 10, 20
        text_label = inp["lang_goal"][0]
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


if __name__ == "__main__":
    main()
