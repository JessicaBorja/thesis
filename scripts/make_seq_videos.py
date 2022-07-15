import cv2
import numpy as np
import os
from glob import glob
import tqdm
from thesis.evaluation.utils import add_text

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
    caption_file = os.path.join(input_dir, "captions.txt")
    with open(caption_file) as f:
        captions = f.readlines()
    return captions

def make_rollout_videos(input_dir):
    fps=30
    task_directories = glob(input_dir + "/*/", recursive=True)
    for task in task_directories:
        policy_dir = glob(task_directories + "/*/", recursive=True)
        for policy in policy_dir:
            cam_type = glob(policy_dir + "/*/", recursive=True)
            for cam in cam_type:
                filename = os.path.join(input_dir, task)
                filename = os.path.join(filename, "%s_%s.mp4" % (cam, policy))
                video_imgs = glob(input_dir + "/*/*.png", recursive=True)
                make_video(video_imgs, fps, filename, caption=policy)
    return

if __name__ == "__main__":
    input_dir = "/mnt/ssd_shared/Users/Jessica/Documents/Thesis_ssd/thesis/hydra_outputs/evaluation_rollouts"
    make_rollout_videos(input_dir)