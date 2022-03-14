import hydra
import cv2
from thesis.env_wrappers.play_lmp_wrapper import PlayLMPWrapper
from thesis.utils.utils import add_img_text, get_abspath, load_aff_model, resize_pixel


@hydra.main(config_path="../config", config_name="cfg_calvin")
def main(cfg):
    # Load affordance model
    checkpoint_path = get_abspath(cfg.aff_checkpoint.train_folder)
    point_detector = load_aff_model(checkpoint_path,
                                    cfg.aff_checkpoint.model_name,
                                    cfg.aff_detection.model,
                                    transforms=cfg.aff_detection.dataset.transforms['validation'])
                                    # hough_voting=cfg.hough_voting)
    point_detector.eval()

    # Load env
    env = hydra.utils.instantiate(cfg.env)
    env = PlayLMPWrapper(env, point_detector.device)
    agent = hydra.utils.instantiate(cfg.agent, env=env, point_detector=point_detector)    

    obs = env.reset()
    for i in range(100):  # n instructions
        rgb_obs =  obs["rgb_obs"]["rgb_static"]
        caption = "use the switch to turn on the light bulb" # input("Type an instruction \n")

        inp = {"img": rgb_obs,
               "lang_goal": caption}

        pred = point_detector.predict(inp)
        point_detector.viz_preds(inp, pred, waitkey=1)
        pixel = resize_pixel(pred["pixel"], pred['softmax'].shape[:2], rgb_obs.shape[:2])

        # World pos
        depth = obs["depth_obs"]["depth_static"]
        world_pos = env.cameras[0].deproject(pixel, depth)
        # p.addUserDebugText("t", textPosition=world_pos, textColorRGB=[1, 0, 1])

        # Rollout
        obs, _, _, info = agent.move_to(world_pos, gripper_action=1)
        goal = agent.encode(caption)
        for j in range(cfg.max_timesteps):
            action = agent.step(obs, goal)
            obs, _, _, info = env.step(action)

        obs, _, _, info = agent.reset_position()

if __name__ == "__main__":
    main()
