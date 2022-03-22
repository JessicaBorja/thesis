from turtle import color
from thesis.agents.base_agent import BaseAgent
from thesis.utils.utils import get_abspath, resize_pixel, pos_orn_to_matrix
from thesis.models.core.language_network import SBert

from calvin_agent.models.play_lmp import PlayLMP
from lfp.datasets.utils.episode_utils import process_depth, process_rgb, process_state, load_dataset_statistics
from omegaconf import DictConfig, OmegaConf
import torch.nn as nn
import os
import torch
import gym.spaces as spaces
import torchvision
import hydra
import cv2

from typing import Any, Dict, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import logging
import importlib

logger = logging.getLogger(__name__)


class PlayLMPAgent(BaseAgent):
    def __init__(self, env, dataset_path, checkpoint=None, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.dataset_path = Path(get_abspath(dataset_path))  # Dataset on which agent was trained
        self.lang_enc = SBert('paraphrase-MiniLM-L3-v2')

        if checkpoint:
            self.model_free, self.transforms = self.load_model_free(**checkpoint)
            self.relative_actions = "rel_actions" in self.observation_space_keys["actions"]
        else:
            self.model_free = PlayLMP()
            self.transforms = nn.Idendity()
            self.relative_actions = True
        self.model_free = self.model_free.to(self.device)
        logger.info(f"Initialized PlayLMPAgent for device {self.device}")

    def instantiate_transforms(self, transforms):
        _transforms = {
            cam: [hydra.utils.instantiate(transform) for transform in transforms[cam]] for cam in transforms
        }
        _transforms = {key: torchvision.transforms.Compose(val) for key, val in _transforms.items()}
        return _transforms  

    def load_model_free(self, train_folder, model_name, **kwargs):
        checkpoint_path = get_abspath(train_folder)
        policy_cfg = os.path.join(checkpoint_path, "./.hydra/config.yaml")
        if os.path.isfile(policy_cfg):
            run_cfg = OmegaConf.load(policy_cfg)
            run_cfg = OmegaConf.create(OmegaConf.to_yaml(run_cfg).replace("calvin_models.", ""))
            checkpoint = os.path.join(checkpoint_path, model_name)
            model_class = run_cfg.model._target_.split('.')
            model_file = '.'.join(run_cfg.model._target_.split('.')[:-1])
            model_file = importlib.import_module(model_file)
            model_class = getattr(model_file, model_class[-1])
            # Parameter added after model was trained
            if 'spatial_softmax_temp' not in run_cfg.model.perceptual_encoder.rgb_static:
                perceptual_encoder = OmegaConf.to_container(run_cfg.model.perceptual_encoder)
                for k in perceptual_encoder.keys():
                    v = perceptual_encoder[k]
                    if isinstance(v, dict) and 'spatial_softmax_temp' not in v and '_target_' in v \
                    and v['_target_'] == 'lfp.models.perceptual_encoders.vision_network.VisionNetwork':
                        perceptual_encoder[k]['spatial_softmax_temp'] = 1.0
                perceptual_encoder = DictConfig(perceptual_encoder)
            model = model_class.load_from_checkpoint(checkpoint, perceptual_encoder=perceptual_encoder)
            model.freeze()
            print("Successfully loaded model.")
            _transforms = run_cfg.datamodule.transforms
            transforms = load_dataset_statistics(self.dataset_path / "training",
                                                 self.dataset_path / "validation",
                                                 _transforms)

            transforms = self.instantiate_transforms(transforms["val"])
            if run_cfg.model.action_decoder.get("load_action_bounds", False):
                model.action_decoder._setup_action_bounds(self.dataset_path, None, None, True)
            env_cfg = run_cfg.datamodule
            self.observation_space_keys = env_cfg.observation_space
            self.proprio_state = env_cfg.proprioception_dims
            _action_min = np.array(env_cfg.action_min)
            _action_high = np.array(env_cfg.action_max)
            self.action_space = spaces.Box(_action_min, _action_high)
        else:
            model = PlayLMP()
            transforms = nn.Idendity()
        return model, transforms

    def transform_observation(self, obs: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        state_obs = process_state(obs, self.observation_space_keys, self.transforms, self.proprio_state)
        rgb_obs = process_rgb(obs["rgb_obs"], self.observation_space_keys, self.transforms)
        depth_obs = process_depth(obs["depth_obs"], self.observation_space_keys, self.transforms)

        state_obs["robot_obs"] = state_obs["robot_obs"].to(self.device).unsqueeze(0)
        rgb_obs.update({"rgb_obs": {k: v.to(self.device).unsqueeze(0) for k, v in rgb_obs["rgb_obs"].items()}})
        depth_obs.update({"depth_obs": {k: v.to(self.device).unsqueeze(0) for k, v in depth_obs["depth_obs"].items()}})

        obs_dict = {**rgb_obs, **state_obs, **depth_obs}
        obs_dict["robot_obs_raw"] = torch.from_numpy(obs["robot_obs"]).to(self.device)
        return obs_dict

    def transform_action(self, action_tensor: torch.Tensor):
        if self.relative_actions:
            action = action_tensor.squeeze().cpu().detach().numpy()
            assert len(action) == 7
        else:
            if action_tensor.shape[-1] == 7:
                slice_ids = [3, 6]
            elif action_tensor.shape[-1] == 8:
                slice_ids = [3, 7]
            else:
                logger.error("actions are required to have length 8 (for euler angles) or 9 (for quaternions)")
                raise NotImplementedError
            action = np.split(action_tensor.squeeze().cpu().detach().numpy(), slice_ids)
        action[-1] = 1 if action[-1] > 0 else -1
        return action

    def encode(self, goal):
        _goal_embd = self.lang_enc(goal).permute(1,0)
        return {'lang': _goal_embd}

    def reset(self, caption):
        self.reset_position()
        self.model_free.reset()

        obs = self.env.get_obs()
        inp = {"img": obs["rgb_obs"]["rgb_static"],
               "lang_goal": caption}
        im_shape = inp["img"].shape[:2]

        pred = self.point_detector.predict(inp)
        # self.point_detector.viz_preds(inp, pred, waitkey=1)

        pixel = resize_pixel(pred["pixel"], pred['softmax'].shape[:2], im_shape)

        # World pos
        depth = obs["depth_obs"]["depth_static"]
        n = 5
        x_range =[max(pixel[0] - n, 0), min(pixel[0] + n, im_shape[1])]
        y_range =[max(pixel[1] - n, 0), min(pixel[1] + n, im_shape[1])]

        target_pos = self.env.cameras[0].deproject(pixel, depth)
        for i in range(x_range[0], x_range[1]):
            for j in range(y_range[0], y_range[1]):
                pos = self.env.cameras[0].deproject((i, j), depth)
                if pos[1] < target_pos[1]:
                    target_pos = pos

        # Add offset
        obs = self.env.get_obs()
        robot_orn = obs['robot_obs'][3:6]
        tcp_mat = pos_orn_to_matrix(target_pos, robot_orn)
        offset_global_frame = tcp_mat @ self.offset
        target_pos = offset_global_frame[:3]

        # img = obs["rgb_obs"]["rgb_static"]
        # pixel = self.env.cameras[0].project(np.array([*target_pos, 1]))
        # img = self.print_px_img(img, pixel)
        # cv2.imshow("move_to", img[:, :, ::-1])
        # cv2.waitKey(1)
        # import pybullet as p
        # p.addUserDebugText("t", target_pos, [1,0,0])

        obs, _, _, info = self.move_to(target_pos, gripper_action=1)

        # Update target pos and orn
        self.env.robot.target_pos = obs["robot_obs"][:3]
        self.env.robot.target_orn = obs["robot_obs"][3:6]

    def print_px_img(self, img, px):
        out_shape = (300, 300)
        pixel = resize_pixel(px, img.shape[:2], out_shape)
        pred_img = img.copy()
        pred_img = cv2.resize(pred_img, out_shape)
        pred_img = cv2.drawMarker(
                pred_img,
                (pixel[0], pixel[1]),
                (0, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        return pred_img

    def step(self, obs, goal_embd):
        '''
            obs(dict):  Observation comming from the environment
                - rgb_obs (dict): 
                    keys:rgb_camName vals: cam_image
                    shape = (C, H, W)
                - depth_obs (dict): keys:depth_camName vals: cam_image
                - robot_obs:
            goal(dict): 
            Either a language or image goal. If language contains key "lang" which is used by the policy to make the prediction, otherwise the goal is provided in the form of an image.
                - lang: caption used to contidion the policy
                Only used if "lang" not in dictionary...
                # B, 384
                - depth_obs: 
                - rgb_obs: 
        '''
        obs = self.transform_observation(obs)
        # imgs: B, S, C, W, H
        action = self.model_free.step(obs, goal_embd)
        action = self.transform_action(action)
        return action
