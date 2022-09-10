from msilib import sequence
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
from robot_io.utils.utils import FpsController

from lfp.evaluation.utils import imshow_tensor
from lfp.models.play_lmp import PlayLMP
from lfp.utils.utils import format_sftp_path, get_checkpoints_for_epochs
from thesis.env_wrappers.aff_lfp_real_world_wrapper import PandaLfpWrapper

from thesis.models.language_encoders.sbert_lang_encoder import SBertLang
import logging
logger = logging.getLogger(__name__)


def load_dataset(cfg):
    train_cfg_path = Path(cfg.train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    train_cfg = OmegaConf.load(train_cfg_path)

    # we don't want to use shm dataset for evaluation
    # since we don't use the trainer during inference, manually set up data_module
    vision_lang_folder = train_cfg.datamodule.datasets.vision_dataset.lang_folder
    lang_lang_folder = train_cfg.datamodule.datasets.lang_dataset.lang_folder
    train_cfg.datamodule.datasets = cfg.datamodule.datasets
    train_cfg.datamodule.datasets.vision_dataset.lang_folder = vision_lang_folder
    train_cfg.datamodule.datasets.lang_dataset.lang_folder = lang_lang_folder

    train_cfg.datamodule.root_data_dir = cfg.datamodule.root_data_dir
    data_module = hydra.utils.instantiate(train_cfg.datamodule, num_workers=0)
    print(train_cfg.datamodule)
    print("data module prepare_data()")
    data_module.prepare_data()
    data_module.setup()
    print("data module setup complete")
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["lang"]

    return dataset, cfg.datamodule.root_data_dir


def evaluate_policy_dataset(model, env, dataset, max_ts, use_affordances):
    i = 0
    print("Press A / D to move through episodes, E / Q to skip 50 episodes.")
    print("Press P to replay recorded actions of the current episode.")
    print("Press O to run inference with the model, but use goal from episode.")
    print("Press L to run inference with the model and use your own language instruction.")
    lang_enc = SBertLang()

    while 1:
        episode = dataset[i]
        imshow_tensor("start", episode["rgb_obs"]["rgb_static"][0], wait=1, resize=True)
        imshow_tensor("start_gripper", episode["rgb_obs"]["rgb_gripper"][0], wait=1, resize=True)
        imshow_tensor("goal_gripper", episode["rgb_obs"]["rgb_gripper"][-1], wait=1, resize=True)
        k = imshow_tensor("goal", episode["rgb_obs"]["rgb_static"][-1], wait=0, resize=True)

        if k == ord("a"):
            i -= 1
            i = int(np.clip(i, 0, len(dataset)))
        if k == ord("d"):
            i += 1
            i = int(np.clip(i, 0, len(dataset)))
        if k == ord("q"):
            i -= 50
            i = int(np.clip(i, 0, len(dataset)))
        if k == ord("e"):
            i += 50
            i = int(np.clip(i, 0, len(dataset)))

        # replay episode with recorded actions
        if k == ord("p"):
            env.reset(episode=episode)
            for action in episode["actions"]:
                env.step(action)
                env.render("human")
        # inference with model, but goal from episode
        if k == ord("o"):
            # env.reset(episode=episode)
            goal = {"lang": episode["lang"].unsqueeze(0).cuda()}
            rollout(env, model, goal)
        # inference with model language prompt
        if k == ord("l"):
            caption = input("Type an instruction \n")
            if model.model_free.lang_encoder is not None:
                goal = {"lang": [caption]}
            else:
                goal = lang_enc.get_lang_goal(caption)
            rollout(env, model, goal, use_affordances, ep_len=max_ts)
            model.save_dir["rollout_counter"] += 1

def rollout(env, model, goal, use_affordances=False, ep_len=340):
    env.reset()
    if use_affordances:
        # width = env.robot.get_observation()[-1]["gripper_opening_width"]
        # if width > 0.055 or width< 0.01:
        model.reset(goal)
    else:
        # If no caption provided, wont use affordance to move to something
        model.model_free.reset()
    obs = env.get_obs()
    model.replan_freq = 15
    for step in range(ep_len):
        action = model.step(obs, goal)
        obs, _, _, _ = env.step(action)
        imshow_tensor("rgb_static", obs["rgb_obs"]["rgb_static"], wait=1, resize=True)
        k = imshow_tensor("rgb_gripper", obs["rgb_obs"]["rgb_gripper"], wait=1, resize=True)
        # press ESC to stop rollout and return
        if k == 27:
            return

@hydra.main(config_path="../../conf", config_name="cfg_real_world")
def main(cfg):
    # load robot
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    dataset, dataset_path = load_dataset(cfg)
    env = PandaLfpWrapper(env, dataset)

    use_affordances = cfg.aff_checkpoint.train_folder is not None
    model = hydra.utils.instantiate(cfg.agent,
                                    dataset_path=dataset_path,
                                    env=env,
                                    use_aff=use_affordances)
    print(f"Successfully loaded affordance model: {cfg.affordance.train_folder}/{cfg.affordance.checkpoint}")
    logger.info(f"Successfully loaded affordance model: {cfg.affordancetrain_folder}/{cfg.affordance.checkpoint}")

    evaluate_policy_dataset(model, env, dataset,
                            use_affordances=use_affordances,
                            max_ts=cfg.max_timesteps)


if __name__ == "__main__":
    main()
