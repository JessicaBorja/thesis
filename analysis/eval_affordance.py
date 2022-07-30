import hydra
from pathlib import Path
from omegaconf import OmegaConf
from thesis.evaluation.utils import get_env_state_for_initial_condition, get_env
import torch

def init_env(cfg, device=0):
    scene = cfg.scene
    if cfg.scene is not None:
        s = "config/scene/%s.yaml" % cfg.scene
        scene = OmegaConf.load(Path(__file__).parents[2] / s)

    camera_conf = cfg.cameras
    if cfg.cameras is not None:
        s = "config/cameras/%s.yaml" % cfg.cameras
        camera_conf = OmegaConf.load(Path(__file__).parents[2] / s)

    # we don't want to use shm dataset for evaluation
    datasets_cfg = cfg.datamodule.datasets
    for k in datasets_cfg.keys():
        datasets_cfg[k]['_target_'] = 'lfp.datasets.npz_dataset.NpzDataset'
        
    cfg.datamodule.datasets = datasets_cfg
    cfg.datamodule.root_data_dir = cfg.dataset_path
    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    dataloader = data_module.val_dataloader()
    dataset = dataloader.dataset.datasets["lang"]
    device = torch.device(f"cuda:{device}")

    dataset = cfg.dataset
    env = get_env(dataset.abs_datasets_dir, show_gui=False, obs_space=dataset.observation_space,
                    scene=scene, camera_conf=camera_conf)
    rollout_cfg = OmegaConf.load(Path(__file__).parents[2] / "config/lfp/rollout/aff_lfp.yaml")
    env = hydra.utils.instantiate(rollout_cfg.env_cfg, env=env, device=device)

def reset_env(self, initial_state):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    self.env.reset(robot_obs=None, scene_obs=scene_obs)


def main():
    env = init_env()

if __name__ == "__main__":
    main()
