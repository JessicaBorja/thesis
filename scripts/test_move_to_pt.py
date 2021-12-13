import hydra
import numpy as np
import cv2
from vr_env.envs.play_table_env import PlayTableSimEnv
import pybullet as p


@hydra.main(config_path="../config", config_name="cfg_calvin")
def main(cfg):
    env = hydra.utils.instantiate(cfg.env)
    agent = hydra.utils.instantiate(cfg.agent, env=env)
    point_detector = hydra.utils.instantiate(cfg.point_detector)
    
    ns = env.reset()
    for i in range(10):  # 5 instructions
        rgb_obs =  ns["rgb_obs"]["rgb_static"]
        cv2.imshow("orig img",rgb_obs[:, :, ::-1])
        cv2.waitKey(1)
        ns = agent.reset_position()
        caption = input("Type an instruction \n")
        pixel = point_detector.find_target({"rgb_obs": rgb_obs,
                                            "caption": caption})
        
        # World pos
        depth = ns["depth_obs"]["depth_static"]
        world_pos = env.cameras[0].deproject(pixel, depth)
        p.addUserDebugText("t", textPosition=world_pos, textColorRGB=[1, 0, 1])
        ns = agent.move_to(world_pos, gripper_action=1)


if __name__ == "__main__":
    main()
