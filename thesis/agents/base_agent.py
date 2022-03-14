import numpy as np
import cv2
import logging
from thesis.affordance.base_detector import BaseDetector
from thesis.utils.utils import add_img_text, resize_pixel
import torch

class BaseAgent:
    def __init__(self, env, point_detector=BaseDetector(), *args, **kwargs):
        self._env = env
        _info = self.env.robot.get_observation()[-1]
        self.origin = np.array(_info["tcp_pos"])
        self.target_orn = np.array(_info["tcp_orn"])
        self.logger = logging.getLogger(__name__)
        _device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(_device)
        self.point_detector=point_detector
        self.model_free = None

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, value):
        self._env = value

    def load_model_free(self, train_folder, model_name, **kwargs):
        self.logger.info("Base Agent has no policy implemented. Step will be a waiting period...")

    def encode(self, goal):
        return goal

    def step(self, obs, goal):
        '''
            obs(dict):  Observation comming from the environment
                - rgb_obs:
                - depth_obs:
                - robot_obs:
            goal(dict): Either a language or image goal. If language contains key "lang" which is used by the policy to make the prediction, otherwise the goal is provided in the form of an image.
                - lang: caption used to contidion the policy
                Only used if "lang" not in dictionary...
                - depth_obs: 
                - rgb_obs: 
        '''
        action = np.zeros(7)
        action[-1] = 1  # gripper
        return action

    def reset_position(self):
        return self.move_to(self.origin,
                            self.target_orn,
                            1)

    def viz_img(rgb_img, center_pixel, output_size=(300, 300), lang_goal=""):
        old_shape = rgb_img.shape[:2]
        pixel = resize_pixel(center_pixel, old_shape, output_size)
        out_img = cv2.drawMarker(
                rgb_img.copy(),
                (pixel[0], pixel[1]),
                (0, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        out_img = cv2.resize(out_img, output_size)
        if(lang_goal != ""):
            # Prints the text.
            out_img = add_img_text(out_img, lang_goal)

        cv2.imshow("orig img",out_img[:, :, ::-1])
        cv2.waitKey(1)

    def move_to(self, target_pos, target_orn=None, gripper_action=None):
        '''
            Move to the specified location in world coordinates
        '''
        curr_info = self.env.robot.get_observation()[-1]
        if(target_orn is None):
            target_orn = np.array(curr_info["tcp_orn"])
        if(gripper_action is None):
            gripper_action = curr_info["gripper_action"]
        action = [target_pos.copy(),
                  target_orn.copy(),
                  gripper_action]
        last_pos = target_pos.copy()
        # When robot is moving and far from target
        ns, r, d, info = self.env.step(action)
        curr_pos = np.array(info["robot_info"]["tcp_pos"])
        while(np.linalg.norm(curr_pos - target_pos) > 0.01 and np.linalg.norm(last_pos - curr_pos) > 0.001):
            last_pos = curr_pos
            ns, r, d, info = self.env.step(action)             
            curr_pos = np.array(info["robot_info"]["tcp_pos"])

            cv2.imshow("obs", ns["rgb_obs"]["rgb_static"][:, :, ::-1])
            cv2.waitKey(1)
        return ns, r, d, info