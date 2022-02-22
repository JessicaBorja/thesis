import hydra
import cv2
import os

from thesis.utils.utils import add_img_text, get_abspath, load_aff_model, resize_pixel

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
    out_img = add_img_text(out_img, lang_goal)
    cv2.imshow("orig img",out_img[:, :, ::-1])
    cv2.waitKey(1)

@hydra.main(config_path="../config", config_name="cfg_calvin")
def main(cfg):
    # # Load policy
    # checkpoint_path = get_abspath(cfg.policy_checkpoint.train_folder)
    # policy_cfg = os.path.join(checkpoint_path, "./.hydra/config.yaml")
    # if os.path.isfile(policy_cfg):
    #     policy_cfg = OmegaConf.load(policy_cfg)
    #     agent_cfg = policy_cfg.model
    # else:
    #     policy_cfg = cfg
    env = hydra.utils.instantiate(cfg.env)
    agent = hydra.utils.instantiate(cfg.agent, env=env)
    
    # Load affordance model
    checkpoint_path = get_abspath(cfg.aff_checkpoint.train_folder)
    point_detector = load_aff_model(checkpoint_path,
                                    cfg.aff_checkpoint.model_name,
                                    cfg.aff_detection.model,
                                    transforms=cfg.aff_detection.dataset.transforms['validation'],
                                    hough_voting=cfg.hough_voting)
    point_detector.eval()

    obs = env.reset()
    for i in range(100):  # n instructions
        rgb_obs =  obs["rgb_obs"]["rgb_static"]
        caption = input("Type an instruction \n")
        inp = {"img": rgb_obs,
               "lang_goal": caption}
        pred = point_detector.predict(inp)
        point_detector.viz_preds(inp, pred)

        # viz_img(rgb_obs, caption, pred, img_input.shape[:2])
        pixel = resize_pixel(pred["pixel"], pred['softmax'].shape[:2], rgb_obs.shape[:2])

        # World pos
        depth = obs["depth_obs"]["depth_static"]
        world_pos = env.cameras[0].deproject(pixel, depth)
        # p.addUserDebugText("t", textPosition=world_pos, textColorRGB=[1, 0, 1])

        # Rollout
        obs, _, _, info = agent.move_to(world_pos, gripper_action=1)
        goal={"lang": caption}
        for j in range(cfg.max_timesteps):
            action = agent.step(obs, goal)
            obs, _, _, info = env.step(action)

        obs, _, _, info = agent.reset_position()

if __name__ == "__main__":
    main()
