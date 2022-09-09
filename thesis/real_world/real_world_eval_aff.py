# from msilib import sequence
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
from robot_io.utils.utils import FpsController

from lfp.evaluation.utils import imshow_tensor
from lfp.models.play_lmp import PlayLMP
from lfp.utils.utils import format_sftp_path, get_checkpoints_for_epochs
from lfp.wrappers.panda_lfp_wrapper import PandaLfpWrapper

from thesis.models.language_encoders.sbert_lang_encoder import SBertLang
import logging
logger = logging.getLogger(__name__)


def load_dataset(cfg):
    train_cfg_path = Path(cfg.train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    train_cfg = OmegaConf.load(train_cfg_path)

    # we don't want to use shm dataset for evaluation
    # since we don't use the trainer during inference, manually set up data_module
    # vision_lang_folder = train_cfg.datamodule.datasets.vision_dataset.lang_folder
    # lang_lang_folder = train_cfg.datamodule.datasets.lang_dataset.lang_folder
    # train_cfg.datamodule.datasets = cfg.datamodule.datasets
    # train_cfg.datamodule.datasets.vision_dataset.lang_folder = vision_lang_folder
    # train_cfg.datamodule.datasets.lang_dataset.lang_folder = lang_lang_folder
    # print(train_cfg.aff_detection.streams.transforms.validation)
    # print(cfg.datamodule.transforms.val.keys())
    #
    # exit()
    cfg.datamodule.transforms.train.rgb_static = train_cfg.aff_detection.streams.transforms.training
    cfg.datamodule.transforms.val.rgb_static = train_cfg.aff_detection.streams.transforms.validation
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    print("data module prepare_data()")
    data_module.prepare_data()
    data_module.setup()
    print("data module setup complete")
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["vis"]

    return dataset, cfg.datamodule.root_data_dir


def evaluate_aff(model, env, max_ts, use_affordances):
    lang_enc = SBertLang()
    while 1:
        # lang_input = [input("What should I do? \n")]
        # lang_embedding = lang_encoder(lang_input)
        # goal = {"lang": lang_embedding.squeeze(0)}
        # print("sleeping 5 seconds...)")
        # time.sleep(6)
        # rollout(env, model, goal)

        goal = input("Type an instruction \n")
        # goal = lang_enc.encode_text([caption])
        rollout(env, model, goal, use_affordances, ep_len=max_ts)
        model.save_dir["rollout_counter"] += 1
        model.rollout()

def rollout(env, model, goal, use_affordances=False, ep_len=340):
    # env.reset()
    # model.reset()
    obs = env.get_obs()
    if use_affordances:
        # width = env.robot.get_observation()[-1]["gripper_opening_width"]
        # if width > 0.055 or width< 0.01:
        target_pos, _move_flag = model.get_aff_pred(obs, goal)

@hydra.main(config_path="../../config", config_name="cfg_real_world")
def main(cfg):
    # load robot
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    dataset, dataset_path = load_dataset(cfg)
    env = PandaLfpWrapper(env, dataset)
    use_affordances = cfg.train_folder is not None
    cfg.agent.aff_cfg.train_folder = cfg.train_folder
    model = hydra.utils.instantiate(cfg.agent,
                                    dataset_path=dataset_path,
                                    env=env,
                                    model_free=False,
                                    use_aff=use_affordances)
    print(f"Successfully loaded affordance model: {cfg.agent.aff_cfg.train_folder}/{cfg.agent.aff_cfg.model_name}")
    logger.info(f"Successfully loaded affordance model: {cfg.agent.aff_cfg.train_folder}/{cfg.agent.aff_cfg.model_name}")

    evaluate_aff(model, env,
                            use_affordances=use_affordances,
                            max_ts=cfg.max_timesteps)


if __name__ == "__main__":
    main()
