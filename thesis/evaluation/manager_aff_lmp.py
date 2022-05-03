import logging
from pathlib import Path

import hydra
from termcolor import colored
from omegaconf import OmegaConf
import torch
import time
from thesis.evaluation.utils import join_vis_lang, format_sftp_path, LangEmbeddings, get_env

logger = logging.getLogger(__name__)


class PolicyManager:
    def __init__(self, debug=False) -> None:
        self.debug = debug

    def rollout(self, env, model, task_oracle, args, subtask, lang_embeddings, val_annotations, plans):
        if args.debug:
            print(f"{subtask} ", end="")
            time.sleep(0.5)
        # get lang annotation for subtask
        lang_annotation = val_annotations[subtask][0]

        # get language goal embedding
        goal = lang_embeddings.get_lang_goal(lang_annotation)

        # Do not reset model if holding something
        width = env.robot.get_observation()[-1]["gripper_opening_width"]
        if width > 0.055 or width< 0.01:
            model.reset(lang_annotation)
        
        # Reset environment
        start_info = env.get_info()
        obs = env.get_obs()
        t_obs = model.transform_observation(obs)
        plan, latent_goal = model.model_free.get_pp_plan_lang(t_obs, goal)
        plans[subtask].append((plan.cpu(), latent_goal.cpu()))

        for step in range(args.ep_len):
            action = model.step(obs, goal)
            obs, _, _, current_info = env.step(action)
            if args.debug:
                img = env.render(mode="rgb_array")
                join_vis_lang(img, lang_annotation)
                # time.sleep(0.1)
            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if args.debug:
                    print(colored("success", "green"), end=" ")
                return True
        if args.debug:
            print(colored("fail", "red"), end=" ")
        return False

    def get_default_model_and_env(self, train_folder, dataset_path, checkpoint, env=None, lang_embeddings=None, device_id=0, scene=None):
        train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
        train_cfg_path = format_sftp_path(train_cfg_path)
        cfg = OmegaConf.load(train_cfg_path)
        cfg = OmegaConf.create(OmegaConf.to_yaml(cfg).replace("calvin_models.", ""))
        lang_folder = cfg.datamodule.datasets.lang_dataset.lang_folder
        if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
            hydra.initialize(config_path="../../config")

        # we don't want to use shm dataset for evaluation
        datasets_cfg = cfg.datamodule.datasets
        for k in datasets_cfg.keys():
            datasets_cfg[k]['_target_'] = 'lfp.datasets.npz_dataset.NpzDataset'

        # since we don't use the trainer during inference, manually set up data_module
        cfg.datamodule.datasets = datasets_cfg
        cfg.datamodule.root_data_dir = dataset_path
        data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
        data_module.prepare_data()
        data_module.setup()
        dataloader = data_module.val_dataloader()
        dataset = dataloader.dataset.datasets["lang"]
        device = torch.device(f"cuda:{device_id}")

        if lang_embeddings is None:
            lang_embeddings = LangEmbeddings(dataset.abs_datasets_dir, lang_folder, device=device)

        if env is None:
            env = get_env(dataset.abs_datasets_dir, show_gui=False, obs_space=dataset.observation_space,
                          scene=scene)
            rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "config/lfp/rollout/aff_lfp.yaml")
            env = hydra.utils.instantiate(rollout_cfg.env_cfg, env=env, device=device)

        checkpoint = format_sftp_path(checkpoint)
        print(f"Loading model from {checkpoint}")

        # Load model model-free + model-based + aff_model from cfg_calvin
        # overwrite model-free from checkpoint
        cfg = hydra.compose(config_name="cfg_calvin")
        cfg.agent.checkpoint.train_folder = train_folder
        cfg.agent.checkpoint.model_name = checkpoint.name
        model = hydra.utils.instantiate(cfg.agent,
                                        viz_obs=self.debug,
                                        env=env,
                                        aff_cfg=cfg.aff_detection,
                                        depth_cfg=cfg.depth_pred)
        print("Successfully loaded model.")

        return model, env, data_module, lang_embeddings
