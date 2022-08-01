from tkinter import N
import cv2
from cv2 import rectangle
import numpy as np
import os
from glob import glob
import tqdm
from thesis.evaluation.utils import add_title
from pathlib import Path


def make_video(files, fps=60, filename="v", caption=""):
    h, w, c = cv2.imread(files[0]).shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(filename, fourcc, fps, (w, h))  # 30 fps
    print("writing video to %s" % filename)
    for f in tqdm.tqdm(files):
        img = cv2.imread(f)
        if caption != "":
            img.add_text(caption)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()

def read_captions(input_dir):
    caption_file = os.path.join(input_dir, "completed_tasks.txt")
    with open(caption_file) as f:
        captions = f.read().splitlines()
    return captions

def add_bottom_txt(img, txt):
    im_w, im_h = img.shape[:2]

    # Add caption
    font_scale = 0.5
    thickness = 1
    x1, y1 = 5, im_h - 20
    color = (255, 255, 255)
    (w, h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    out_img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1 + h), color, -1)

    (w_txt, h_txt), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    coord = (x1, y1)

    out_img = cv2.putText(
        out_img,
        txt,
        org=coord,
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=font_scale,
        color=(0, 0, 0),
        thickness=thickness,
        lineType=cv2.LINE_AA
    )
    return out_img

def merge_images(aff_pred, static_cam_imgs, gripper_cam_imgs, caption, policy_type):
    h, w = static_cam_imgs[0].shape[:2]

    n_pad = 10
    c = 255
    img_list = []
    for static_cam, gripper_cam in zip(static_cam_imgs, gripper_cam_imgs):
        static_img = add_bottom_txt(static_cam, policy_type)
        n_size = (h//2 - n_pad, h//2 - n_pad//2)
        # Left padding
        aff = cv2.resize(aff_pred, n_size)
        aff = np.pad(aff, ((0, n_pad//2), (0, n_pad), (0, 0)), mode='constant', constant_values=c)

        # left and top pad
        g_cam = cv2.resize(gripper_cam, n_size)
        g_cam = np.pad(g_cam, ((n_pad//2, 0), (0, n_pad), (0, 0)), mode='constant', constant_values=c)

        # Form full image
        full_img = np.vstack([aff, g_cam])
        full_img = np.hstack([full_img, static_img])
        
        # Add caption as title
        full_img = add_title(full_img, caption)
        cv2.imshow("x", full_img)
        cv2.waitKey(0)
        img_list.append(full_img)
    return img_list


def make_rollout_videos(input_dir):
    fps=30
    policy_title = {"model_based": "Model-based policy",
                    "model_free": "Learning-based policy"}

    sequences = sorted(glob(input_dir + "/*/", recursive=True))
    for seq_dir in sequences:
        seq_id = int(Path(seq_dir).name.split("_")[-1])
        tasks = sorted(glob(seq_dir + "/*/", recursive=True))
        captions = read_captions(seq_dir)

        for caption, task_dir in zip(captions, tasks):
            policies = sorted(glob(task_dir + "/*/", recursive=True))
            aff_pred = glob(task_dir + "aff_pred*.png", recursive=True)[0]
            aff_img = cv2.imread(aff_pred)
            rollout_imgs = []
            for policy_dir in policies:
                policy_type = Path(policy_dir).name
                cam_folder = glob(policy_dir + "/*/", recursive=True)
                cam_imgs = {}
                for cam_dir in cam_folder:
                    cam = Path(cam_dir).name
                    cam_imgs[cam] = [cv2.imread(x) for x in sorted(glob(cam_dir + "*.png", recursive=True))]

                # create a single images with all images for video
                merged_imgs = merge_images(aff_img,
                                           cam_imgs["static_cam"],
                                           cam_imgs["gripper_cam"],
                                           caption,
                                           policy_title[policy_type])
                rollout_imgs.extend(merged_imgs)
                
    
    # make_video(video_imgs, fps, filename, caption=caption)
    return

if __name__ == "__main__":
    input_dir = "~/logs/evaluation_rollouts/2022-07-30_21-25-24/"
    input_dir = os.path.expanduser(input_dir)
    make_rollout_videos(input_dir)