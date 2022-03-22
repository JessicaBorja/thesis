import hydra
import cv2
from thesis.env_wrappers.play_lmp_wrapper import PlayLMPWrapper
import torch

@hydra.main(config_path="../config", config_name="cfg_calvin")
def main(cfg):
    # Load env
    env = hydra.utils.instantiate(cfg.env)
    env = PlayLMPWrapper(env, torch.device('cuda:0'))
    agent = hydra.utils.instantiate(cfg.agent, env=env, aff_cfg=cfg.aff_detection)    

    obs = env.reset()
    captions = ["Open the drawer",
                "Move the sliding door",
                "Turn on the light switch",
                "Turn on the green led",
                "Lift the pink block"]
    for caption in captions:  # n instructions
        rgb_obs =  obs["rgb_obs"]["rgb_static"]
        # caption = "use the switch to turn on the light bulb" # input("Type an instruction \n")
        # caption = "open the drawer"

        inp = {"img": rgb_obs,
               "lang_goal": caption,
               "depth": obs["depth_obs"]["depth_static"]}

        agent.reset(inp)
        goal = agent.encode(caption)
        for j in range(cfg.max_timesteps):
            action = agent.step(obs, goal)
            obs, _, _, info = env.step(action)
            img = cv2.resize(obs['rgb_obs']['rgb_static'][:, :, ::-1], (300, 300))
            cv2.imshow("static_cam", img)
            cv2.waitKey(1)

        obs, _, _, info = agent.reset_position()

if __name__ == "__main__":
    main()
