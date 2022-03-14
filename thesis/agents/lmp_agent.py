from thesis.agents.base_agent import BaseAgent
from thesis.utils.utils import get_abspath
from thesis.models.language_encoders.language_network import SBert

from calvin_agent.models.play_lmp import PlayLMP
from calvin_agent.datasets.utils.episode_utils import process_depth, process_rgb, process_state
from calvin_agent.datasets.utils.episode_utils import load_dataset_statistics
from omegaconf import OmegaConf
import torch.nn as nn
import os
import torch
import gym.spaces as spaces
import torchvision
import hydra

from typing import Any, Dict, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class PlayLMPAgent(BaseAgent):
    def __init__(self, env, dataset_path, checkpoint=None, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.dataset_path = Path(get_abspath(dataset_path))  # Dataset on which agent was trained
        self.lang_enc = SBert(nlp_model='mini')

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
            model = PlayLMP.load_from_checkpoint(checkpoint)
            model.freeze()
            # if cfg.model.action_decoder.get("load_action_bounds", False):
            #     model.action_decoder._setup_action_bounds(cfg.datamodule.root_data_dir, None, None, True)
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

        return {**rgb_obs, **state_obs, **depth_obs}

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
        return _goal_embd

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
        # for k in ['rgb_obs', 'depth_obs']:
        #     for cam in obs[k].keys():
        #         # batch, seq, channels, H, W
        #         obs[k][cam] = torch.tensor(obs[k][cam]).to(self.device)
        #         if len(obs[k][cam].shape) > 2:
        #             # rgb_obs =  (C, H, W)
        #             obs[k][cam] = obs[k][cam].permute((2, 0, 1))
        #         obs[k][cam] = obs[k][cam].unsqueeze(0).unsqueeze(0)
        # obs['depth_obs']['depth_static'] = None
        # imgs: B, S, C, W, H
        action = self.model_free.step(obs, {"lang": goal_embd})
        action = self.transform_action(action)
        return action
