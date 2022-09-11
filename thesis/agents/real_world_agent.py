from thesis.utils.utils import get_abspath, resize_pixel
from thesis.evaluation.utils import join_vis_lang
from thesis.utils.utils import add_img_text, resize_pixel, get_aff_model

from thesis.utils.episode_utils import load_dataset_statistics, process_depth, process_rgb, process_state
from omegaconf import DictConfig, OmegaConf
import torch.nn as nn
import os
import torch
import gym.spaces as spaces
import torchvision
import hydra
import cv2

from typing import Any, Dict, Union
from pathlib import Path
import numpy as np
import torch
import logging
import importlib
logger = logging.getLogger(__name__)


class AffHULCAgent():
    def __init__(self, env, dataset_path, offset, aff_cfg, model_free=None, move_outside=True,
                 viz_obs=False, save_viz=False, *args, **kwargs):
        self.env = env
        self.device = "cuda"
        # Model-based params
        self.move_outside = move_outside
        self.offset = np.array([*offset, 1])
        self.T_world_cam = self.env.env.camera_manager.static_cam.get_extrinsic_calibration("panda")
        # Revisar que esta orientacion este bien
        self.target_orn = np.array([-3.11,  0.047,  0.027])

        ## Aff modeland language cnc cfg
        self.dataset_path = Path(get_abspath(dataset_path))  # Dataset on which agent was trained
        logger.info("PlayLMPAgent dataset_path: %s" % self.dataset_path)
        if model_free:
            self.model_free, self.transforms = self.load_model_free(**model_free)
            self.relative_actions = "rel_actions" in self.observation_space_keys["actions"]
            self.model_free = self.model_free.to(self.device)


        # Cameras
        self.static_cam = self.env.camera_manager.static_cam
        
        # Not save first
        self.save_viz = False
        self.reset_position()

        # Load Aff model
        _point_detector, _ = get_aff_model(**aff_cfg)
        if _point_detector is not None:
            self.point_detector = _point_detector.to(self.device)
            self.point_detector.model = self.point_detector.model.to(self.device)

        ## Save images
        self.save_viz = save_viz
        self.viz_obs = viz_obs        
        model = "ours" if kwargs["use_aff"] else "baseline"
        save_directory = "./%s_rollouts" % model
        self.save_dir = {"parent": save_directory,
                         "sequence_counter": 0,
                         "rollout_counter": 0,
                         "step_counter": 0}
        self.sequence_data = {}

    def move_to(self, abs_pos, gripper_action=1):
        gripper_state = "open" if gripper_action == 1 else "closed"
        self.env.reset(abs_pos, self.target_orn, gripper_action)

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
            checkpoint = os.path.join(checkpoint_path, "saved_models")

            if isinstance(model_name, int):
                model_name = "epoch=%d.ckpt" % model_name
            checkpoint = os.path.join(checkpoint, model_name)
            model_class = run_cfg.model._target_.split('.')
            model_file = '.'.join(run_cfg.model._target_.split('.')[:-1])
            model_file = importlib.import_module(model_file)
            model_class = getattr(model_file, model_class[-1])
            # Parameter added after model was trained
            if ('rgb_static' in run_cfg.model.perceptual_encoder and
                'spatial_softmax_temp' not in run_cfg.model.perceptual_encoder.rgb_static):
                perceptual_encoder = OmegaConf.to_container(run_cfg.model.perceptual_encoder)
                for k in perceptual_encoder.keys():
                    v = perceptual_encoder[k]
                    if isinstance(v, dict) and 'spatial_softmax_temp' not in v and '_target_' in v \
                    and v['_target_'] == 'lfp.models.perceptual_encoders.vision_network.VisionNetwork':
                        perceptual_encoder[k]['spatial_softmax_temp'] = 1.0
                perceptual_encoder = DictConfig(perceptual_encoder)
                model = model_class.load_from_checkpoint(checkpoint, perceptual_encoder=perceptual_encoder)
            else:
                model = model_class.load_from_checkpoint(checkpoint)
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
            return
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

    def add_offset(self, pos):
        # Add offset
        # obs = self.env.get_obs()
        # robot_orn = obs['robot_obs'][3:6]
        # tcp_mat = pos_orn_to_matrix(pos, robot_orn)
        # offset_global_frame = tcp_mat @ self.offset
        # offset_pos = offset_global_frame[:3]
        offset_pos = pos + self.offset[:3]
        return offset_pos

    def get_aff_pred(self, caption, obs, out_shape=None):
        inp = {"img": obs["rgb_obs"]["rgb_static"],
               "lang_goal": caption}
        im_shape = inp["img"].shape[:2]
        pred = self.point_detector.predict(inp)

        if out_shape is None:
            inp["img"].shape[:2]

        out_img, _info = self.point_detector.get_preds_viz(inp, pred,
                                                           out_shape=out_shape)

        if self.viz_obs:
            cv2.imshow("img", out_img[:, :, ::-1])
            # cv2.imshow("heatmap", (_info["heatmap"])[:, :, ::-1])
            # cv2.imshow("pred_pixel", (_info["pred_pixel"])[:, :, ::-1])
            cv2.waitKey(2)
        
        if self.save_viz:
            heatmap = _info["heatmap"] * 255
            self.save_img(heatmap, ".", "aff_pred")
            self.save_img(_info["pred_pixel"] * 255, ".", "pred_pixel")
            self.save_img(inp["img"], ".", "orig_img")
            self.save_sequence_txt("task", caption)
        pixel = resize_pixel(pred["pixel"], pred['softmax'].shape[:2], im_shape)

        # World pos
        depth = obs["depth_obs"]["depth_static"]
        n = 5
        x_range =[max(pixel[0] - n, 0), min(pixel[0] + n, im_shape[1])]
        y_range =[max(pixel[1] - n, 0), min(pixel[1] + n, im_shape[1])]

        resize_res = np.array(self.static_cam.get_resize_res())
        resize_res = resize_pixel(pred["pixel"], pred['softmax'].shape[:2], resize_res)
        if "depth" in pred:
            depth_sample = pred['depth']
            target_pos = self.static_cam.deproject(resize_res, depth_sample.item())
        else:
            target_pos = self.static_cam.deproject(resize_res, depth)
            for i in range(x_range[0], x_range[1]):
                for j in range(y_range[0], y_range[1]):
                    pos = self.static_cam.deproject((i, j), depth)
                    if pos[1] < target_pos[1]:
                        target_pos = pos

        world_pt = self.T_world_cam @ np.array([*target_pos, 1])
        world_pt = world_pt[:3]
        print("real word point: ", world_pt)
        # offset_pos = world_pt
        offset_pos = self.add_offset(world_pt)

        # If far from target 3d
        # diff_target = np.linalg.norm(target_pos - robot_obs[:3])
        # diff_offset = np.linalg.norm(offset_pos - robot_obs[:3])

        # 2d dist
        tcp_px = self.static_cam.project(np.array([*obs["robot_obs"][:3], 1]))
        px_dist = np.linalg.norm(pixel - tcp_px)
        print("px_dist: ", px_dist)
        print(pixel)
        print(tcp_px)
        move = px_dist > 15
        print("move: ", move)

        # img = obs["rgb_obs"]["rgb_static"]
        # pixel = self.static_cam.project(np.array([*target_pos, 1]))
        # img = self.print_px_img(img, pixel)
        # cv2.imshow("move_to", img[:, :, ::-1])
        # cv2.waitKey(1)
        return offset_pos, move

    def reset_position(self):
        self.env.reset()

    def reset(self, caption):
        self.curr_caption = caption
        self.save_dir["step_counter"] = 0
        if self.move_outside:
            self.reset_position()

        # Open gripper
        obs = self.env.get_obs()
        robot_obs = obs["robot_obs"]
        width = robot_obs[6]
        # Stay in place but open gripper
        if width < 0.03:
            self.env.reset(robot_obs[:3], robot_obs[3:6], "open")

        # Get Target
        offset_pos, move = self.get_aff_pred(caption, obs)
    
        if move:
            obs, _, _, info = self.move_to(offset_pos, gripper_action=1)
            # self.env.robot.target_pos = offset_pos.copy()
            # self.env.robot.target_orn = self.target_orn.copy()

        self.model_free.reset()

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
        '''   
        if self.viz_obs:
            _caption = "MF: %s" % goal_embd['lang'][0]
            join_vis_lang(obs['rgb_obs']['rgb_static'], _caption)
            # img = cv2.resize([:, :, ::-1], (300,300))
            # cv2.imshow("static_cam", img)
            cv2.waitKey(1)   
        if self.save_viz:
            self.save_img(obs["rgb_obs"]["rgb_static"], "./model_free/static_cam")
            self.save_img(obs["rgb_obs"]["rgb_gripper"], "./model_free/gripper_cam")
            self.save_dir["step_counter"] += 1

        obs = self.transform_observation(obs)
        # imgs: B, S, C, W, H
        action = self.model_free.step(obs, goal_embd)
        action = self.transform_action(action)
        return action

    def save_sequence_txt(self, filename, data):
        output_dir = os.path.join(self.save_dir["parent"],
                              "seq_%03d" % self.save_dir["sequence_counter"])
        filedir = os.path.join(output_dir, "%s.txt" % filename)
        if filedir in self.sequence_data:
            self.sequence_data[filedir].append(data)
        else:
            if isinstance(data, list):
                self.sequence_data[filedir] = data
            else:
                self.sequence_data[filedir] = [data]

    def save_img(self, img, folder="./", name="img"):
        outdir = os.path.join(self.save_dir["parent"],
                              "seq_%03d" % self.save_dir["sequence_counter"])
        outdir = os.path.join(outdir,
                              "task_%02d" % self.save_dir["rollout_counter"])
        outdir = os.path.join(outdir, folder)
        output_file = os.path.join(outdir, "%s_%04d" % (name, self.save_dir["step_counter"]))
        self.sequence_data["%s.png" % output_file] = img[:, :, ::-1]
        return output_file

    def save_rollout(self):
        for filename, data in self.sequence_data.items():
            dirname = os.path.dirname(filename)
            os.makedirs(dirname, exist_ok=True)
            if filename.split('.')[-1] == "txt":
                with open(filename, 'w') as f:
                    for line in data:
                        f.write(line)
                        f.write('\n')
            else:
                cv2.imwrite(filename, data)
        self.sequence_data = {}

    # def img_save_viz_mb(self, ns):
    #     if self.viz_obs:
    #         _caption = "MB: %s" % self.curr_caption
    #         join_vis_lang(ns['rgb_obs']['rgb_static'], _caption)
    #         # img = cv2.resize([:, :, ::-1], (300,300))
    #         # cv2.imshow("static_cam", img)
    #         cv2.waitKey(1)

    #     if self.save_viz:
    #         self.save_img(ns["rgb_obs"]["rgb_static"], "./model_based/static_cam")
    #         self.save_img(ns["rgb_obs"]["rgb_gripper"], "./model_based/gripper_cam")
    #         self.save_dir["step_counter"] += 1