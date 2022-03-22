import numpy as np
import cv2
import logging
from thesis.affordance.base_detector import BaseDetector
from thesis.utils.utils import add_img_text, resize_pixel
from thesis.utils.utils import get_abspath, load_aff_model, resize_pixel
import torch
import os

class BaseAgent:
    def __init__(self, env, offset, aff_cfg, viz_obs=False, *args, **kwargs):
        self._env = env
        self.viz_obs = viz_obs
        _info = self.env.robot.get_observation()[-1]
        self.origin = np.array([-0.25, -0.3, 0.6])  # np.array(_info["tcp_pos"])
        self.target_orn = np.array(_info["tcp_orn"])
        self.logger = logging.getLogger(__name__)
        self.point_detector = self.get_point_detector(aff_cfg)
        self.device = self.env.device
        self.model_free = None
        self.offset = np.array([*offset, 1])
        self.reset_position()

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, value):
        self._env = value

    def get_point_detector(self, aff_cfg):
        checkpoint_path = get_abspath(aff_cfg.checkpoint.train_folder)
        if os.path.exists(checkpoint_path):
            point_detector = load_aff_model(checkpoint_path,
                                            aff_cfg.checkpoint.model_name,
                                            aff_cfg.model,
                                            transforms=aff_cfg.dataset.transforms['validation'])
                                            # hough_voting=cfg.hough_voting)
            point_detector.eval()
        else:
            point_detector = BaseDetector()
        return point_detector


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

        
        tcp_pos = np.array(curr_info["tcp_pos"])
        # Move up
        reach_target = [*tcp_pos[:2], self.origin[-1]]
        a = [reach_target.copy(), target_orn.copy(), gripper_action]
        tcp_pos, _ = self.move_to_pos(tcp_pos, a)

        # Move in xy
        reach_target = [*target_pos[:2], tcp_pos[-1]]
        a = [reach_target.copy(), target_orn.copy(), gripper_action]
        tcp_pos, _ = self.move_to_pos(tcp_pos, a)

        # Move to target
        a = [target_pos, target_orn, gripper_action]
        _, transition = self.move_to_pos(tcp_pos, a)
        return transition

    def move_to_pos(self, tcp_pos, action):
        last_pos = tcp_pos.copy()
        target_pos = action[0]
        # When robot is moving and far from target
        ns, r, d, info = self.env.step(action)
        curr_pos = np.array(info["robot_info"]["tcp_pos"])
        while(np.linalg.norm(curr_pos - target_pos) > 0.01
              and np.linalg.norm(curr_pos - last_pos) > 0.001):
            last_pos = curr_pos
            ns, r, d, info = self.env.step(action)             
            curr_pos = np.array(info["robot_info"]["tcp_pos"])

            if self.viz_obs:
                img = cv2.resize(ns['rgb_obs']['rgb_static'][:, :, ::-1], (300,300))
                cv2.imshow("static_cam", img)
                cv2.waitKey(1)
        return curr_pos, (ns, r, d, info)