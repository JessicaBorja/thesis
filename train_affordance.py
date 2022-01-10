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
    
    checkpoint_path = os.path.join(cfg.train_dir, 'checkpoints')
    last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    last_checkpoint = last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg.load_from_last_ckpt else None

    # Initialize model
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="%s_{epoch:02d}" % cfg.run_name,
        **cfg.wandb.saver
    )

    # Trainer
    trainer = Trainer(
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=last_checkpoint,
        **cfg.trainer
    )

    # Resume epoch and global_steps
    if last_checkpoint:
        print(f"Resuming: {last_checkpoint}")
        last_ckpt = torch.load(last_checkpoint)
        trainer.current_epoch = last_ckpt['epoch']
        trainer.global_step = last_ckpt['global_step']
        del last_ckpt

    # Dataloaders
    train = CalvinDataLang(split="training", log=logger, **cfg.dataset)
    val = CalvinDataLang(split="validation", log=logger, **cfg.dataset)
    logger.info("train_data {}".format(train.__len__()))
    logger.info("val_data {}".format(val.__len__()))

    train_loader = DataLoader(train, shuffle=True, **cfg.dataloader)
    val_loader = DataLoader(val, **cfg.dataloader)
    logger.info("train minibatches {}".format(len(train_loader)))
    logger.info("val minibatches {}".format(len(val_loader)))

    # Initialize agent
    model = hydra.utils.instantiate(cfg.aff_detection)

    # Main training loop
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
