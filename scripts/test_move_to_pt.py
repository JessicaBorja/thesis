import hydra
import numpy as np
import cv2
import pybullet as p
import pytorch_lightning as pl
import time
from thesis.utils.utils import get_hydra_launch_dir, load_aff_model, resize_pixel

def viz_img(rgb_img, lang_goal, pred, old_shape):
    pixel = pred["pixel"]
    pixel = resize_pixel(pixel, old_shape, rgb_img.shape[:2])
    out_img = cv2.drawMarker(
            rgb_img.copy(),
            (pixel[0], pixel[1]),
            (0, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
    out_img = cv2.resize(out_img, (300, 300))
    # Prints the text.
    font_scale = 0.6
    thickness = 2
    color = (0, 0, 0)
    x1, y1 = 10, 20
    (w, h), _ = cv2.getTextSize(lang_goal, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    out_img = cv2.rectangle(out_img, (x1, y1 - 20), (x1 + w, y1 + h), color, -1)
    out_img = cv2.putText(
        out_img,
        lang_goal,
        org=(x1, y1),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=thickness,
    )
    cv2.imshow("orig img",out_img[:, :, ::-1])
    cv2.waitKey(1)

@hydra.main(config_path="../config", config_name="cfg_calvin")
def main(cfg):
    env = hydra.utils.instantiate(cfg.env)
    agent = hydra.utils.instantiate(cfg.agent, env=env)
    point_detector = hydra.utils.instantiate(cfg.aff_detection)
    
    # Load model
    checkpoint_path = get_hydra_launch_dir(cfg.aff_checkpoint.path)
    point_detector = load_aff_model(checkpoint_path,
                                    cfg.aff_checkpoint.model_name,
                                    cfg.aff_detection)
    point_detector.eval()
    im_size = cfg.aff_checkpoint.img_resize

    ns = env.reset()
    for i in range(10):  # 5 instructions
        rgb_obs =  ns["rgb_obs"]["rgb_static"]
        caption = input("Type an instruction \n")
        img_input = cv2.resize(rgb_obs, (im_size, im_size))
        pred = point_detector.predict({"img": img_input,
                                       "lang_goal": caption})
        
        viz_img(rgb_obs, caption, pred, img_input.shape[:2])
        pixel = resize_pixel(pred["pixel"], img_input.shape[:2], rgb_obs.shape[:2])
        # World pos
        depth = ns["depth_obs"]["depth_static"]
        world_pos = env.cameras[0].deproject(pixel, depth)
        # p.addUserDebugText("t", textPosition=world_pos, textColorRGB=[1, 0, 1])
        ns = agent.move_to(world_pos, gripper_action=1)
        time.sleep(1)
        ns = agent.reset_position()

if __name__ == "__main__":
    main()
