import os
import json
import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import logging
from thesis.utils.utils import add_img_text, resize_pixel, get_abspath, get_transforms
from thesis.datasets.transforms import NormalizeInverse

class PixeLabelDataLang(Dataset):
    def __init__(
        self,
        img_resize,
        data_dir,
        transforms,
        n_train_ep=-1,
        split="training",
        cam="static",
        log=None,
        episodes_file="episodes_split.json",
        *args,
        **kwargs
    ):
        super(PixeLabelDataLang, self).__init__()
        self.cam = cam
        self.split = split
        self.log = log
        self.data_dir = get_abspath(data_dir)
        _data_info = self.read_json(os.path.join(self.data_dir, episodes_file))
        self.data = self._get_split_data(_data_info, split, cam, n_train_ep)
        self.img_resize = img_resize
        _transforms_dct = get_transforms(transforms[split], img_resize[cam])
        self.transforms = _transforms_dct["transforms"]
        self.rand_shift = _transforms_dct["rand_shift"]
        _norm_vals = _transforms_dct["norm_values"]
        if(_norm_vals is not None):
            self.norm_inverse = NormalizeInverse(mean=_norm_vals.mean, std=_norm_vals.std)
        else:
            self.norm_inverse = None
        self.out_shape = self.get_shape(img_resize[cam])

        # Excludes background
        self.n_classes = _data_info["n_classes"] if cam == "static" else 1
        self.resize = (self.img_resize[self.cam], self.img_resize[self.cam])
        self.cmd_log = logging.getLogger(__name__)
        self.cmd_log.info("Dataloader using shape: %s" % str(self.resize))

    def undo_normalize(self, x):
        if(self.norm_inverse is not None):
            x = self.norm_inverse(x)
        return x

    def read_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data

    def get_shape(self, in_size):
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
        # tcp_world = data['tcp_pos_world_frame']
        old_shape = data["frame"].shape[:2]
        frame = data["frame"]
        orig_frame = torch.from_numpy(frame).permute(2, 0, 1)  # C, W, H
        frame = self.transforms(orig_frame.float())

        # Aff mask
        # data["centers"] = (label, x, y)
        center = resize_pixel(data["centers"][0, 1:], old_shape, self.resize)
        assert (center < self.resize).all(), "center is out of range, old_shape %s, resize %s, old_center %s, new_center %s" % (str(old_shape), str(self.resize), str(data["centers"][0, 1:]), str(center))
        # mask = np.zeros(self.resize)
        # mask[center[0], center[1]] = 1

        # Apply rand shift
        if(self.rand_shift is not None):
            frame, center = self.rand_shift({"img": frame,
                                             "center": center})

        # Select a language annotation
        annotations = [i.item() for i in data["lang_ann"]]
        assert len(annotations) > 0, "no language annotation in %s" % self.data[idx]
        lang_ann = np.random.choice(annotations).item()

        task = data["task"].tolist()
        inp = {"img": frame,  # RGB
               "lang_goal": lang_ann,
               "orig_frame": orig_frame.float() / 255}
        
        # Cam point in -z direction, but depth should be positive
        tcp_cam = data['tcp_pos_cam_frame']
        depth = tcp_cam[-1] * -1  

        # CE Loss requires mask in form (B, H, W)
        labels = {"task": task,
                  "p0": center,
                  "depth": depth,
                  "tetha0": []}
        return inp, labels


@hydra.main(config_path="../../config", config_name="train_affordance")
def main(cfg):
    data = PixeLabelDataLang(split="training", log=None, **cfg.aff_detection.dataset)
    loader = DataLoader(data, num_workers=1, batch_size=1, pin_memory=True)
    print("minibatches {}".format(len(loader)))

    cm = plt.get_cmap("jet")
    colors = cm(np.linspace(0, 1, data.n_classes))
    for b_idx, b in enumerate(loader):
        # RGB
        inp, labels = b
    
        # Imgs to numpy
        inp_img = data.undo_normalize(inp["img"]).detach().cpu().numpy()
        inp_img = (inp_img[0] * 255).astype("uint8")
        transformed_img = np.transpose(inp_img, (1, 2, 0)).copy()

        orig_img = inp["orig_frame"]
        frame = orig_img[0].detach().cpu().numpy()
        frame = (frame * 255).astype("uint8")
        frame = np.transpose(frame, (1, 2, 0))

        frame = cv2.resize(frame, inp["img"].shape[-2:])
        if frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        out_img = frame.copy()
        for label in range(0, data.n_classes):
            color = colors[label]
            color[-1] = 0.3
            color = tuple((color * 255).astype("int32"))

            # Draw center
            center_px = labels["p0"][0].numpy().squeeze()
            y, x = center_px[0].item(), center_px[1].item()
            transformed_img = cv2.drawMarker(
                transformed_img,
                (x, y),
                (0, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

        transformed_img = cv2.resize(transformed_img, (400, 400), interpolation=cv2.INTER_CUBIC)
        out_img = cv2.resize(out_img, (400, 400), interpolation=cv2.INTER_CUBIC)

        out_img = np.hstack([out_img, transformed_img])
        # Prints the text.
        out_img = add_img_text(out_img, inp["lang_goal"][0])

        out_img = out_img[:, :, ::-1]
        if(cfg.save_viz):
            file_dir = "./imgs"
            os.makedirs(file_dir, exist_ok=True)
            filename = os.path.join(file_dir, "frame_%04d.png" % b_idx)
            cv2.imwrite(filename, out_img)
        cv2.imshow("img", out_img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
