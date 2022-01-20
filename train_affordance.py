"""Main training script."""

import os

import torch
from thesis.datasets.calvin_data import CalvinDataLang, DataLoader

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import logging
from omegaconf import OmegaConf
from thesis.utils.utils import get_hydra_launch_dir


def print_cfg(cfg):
    print_cfg = OmegaConf.to_container(cfg)
    print_cfg.pop("dataset")
    print_cfg.pop("trainer")
    return OmegaConf.create(print_cfg)


@hydra.main(config_path="./config", config_name='train_affordance')
def main(cfg):
    # Log main config for debug
    print("Running configuration: ", cfg)
    logger = logging.getLogger(__name__)
    logger.info("Running configuration: %s", OmegaConf.to_yaml(print_cfg(cfg)))

    # Logger
    wandb_logger = WandbLogger(**cfg.wandb.logger)

    # Checkpoint saver
    checkpoint_dir = get_hydra_launch_dir(cfg.checkpoint.path)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_path, cfg.checkpoint.model_name)
    last_checkpoint = checkpoint_path if os.path.exists(checkpoint_path) and cfg.load_from_last_ckpt else None

    # Initialize checkpoints
    callbacks = []
    for name, saver_cfg in cfg.wandb.saver.items():
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="%s_{epoch:02d}_%s" % (cfg.run_name, name),
            **saver_cfg
        )
        callbacks.append(checkpoint_callback)
    # Trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        **cfg.trainer
    )

    # Dataloaders
    train = CalvinDataLang(split="training", log=logger, **cfg.dataset)
    val = CalvinDataLang(split="validation", log=logger, **cfg.dataset)
    logger.info("train_data {}".format(train.__len__()))
    logger.info("val_data {}".format(val.__len__()))

    train_loader = DataLoader(train, shuffle=True, **cfg.dataloader)
    val_loader = DataLoader(val, shuffle=True, **cfg.dataloader)
    logger.info("train minibatches {}".format(len(train_loader)))
    logger.info("val minibatches {}".format(len(val_loader)))

    # Initialize agent
    in_shape = train.out_shape[::-1]  # H, W, C
    model = hydra.utils.instantiate(cfg.aff_detection.model, in_shape=in_shape)

    # Resume epoch and global_steps
    if last_checkpoint:
        print(f"Resuming: {last_checkpoint}")
        model = model.load_from_checkpoint(last_checkpoint).cuda()
        logger.info("Model successfully loaded: %s" % last_checkpoint)

    # Main training loop
    trainer.fit(model, train_loader, val_loader, ckpt_path=last_checkpoint)


if __name__ == '__main__':
    main()
