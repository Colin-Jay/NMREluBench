import argparse
import datetime
from typing import Any, List
import typing as T
from pathlib import Path
import logging
import os
import wandb

from rdkit import RDLogger
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from pytorch_lightning.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning import LightningModule, Trainer, seed_everything, Callback

import omegaconf
from omegaconf import DictConfig, OmegaConf

import src.utils as utils

from src.common.utils import log_hyperparameters, PROJECT_ROOT


class ConsoleLogging(Callback):

    def __init__(self,):
        super().__init__()
        self.logger = logging.getLogger("ConsoleLogging")

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if trainer.local_rank==0 and batch_idx % 500 == 0:
            met = trainer.callback_metrics
            epoch_int = trainer.current_epoch
            msg = ""
            for k, v in met.items():
                if "loss" in k:
                    msg += f'{k}: {v:.4f}, '
            self.logger.info(f'Epoch {epoch_int}, batch {batch_idx}: {msg}')


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    if "lr_monitor" in cfg.logger:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logger.lr_monitor.logging_interval,
                log_momentum=cfg.logger.lr_monitor.log_momentum,
            )
        )

    if "early_stopping" in cfg.callbacks:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.callbacks.monitor_metric,
                mode=cfg.callbacks.monitor_metric_mode,
                patience=cfg.callbacks.early_stopping.patience,
                verbose=cfg.callbacks.early_stopping.verbose,
            )
        )

    if "model_checkpoints" in cfg.callbacks:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(HydraConfig.get().run.dir),
                monitor=cfg.callbacks.monitor_metric,
                mode=cfg.callbacks.monitor_metric_mode,
                save_top_k=cfg.callbacks.model_checkpoints.save_top_k,
                verbose=cfg.callbacks.model_checkpoints.verbose,
                save_last=cfg.callbacks.model_checkpoints.save_last,
                filename="{epoch}-{val_loss:.2f}",
            )
        )
    
    callbacks.append(ConsoleLogging())

    return callbacks

def run(cfg: DictConfig) -> None:
    # Seed everything
    seed_everything(cfg.model.random_seed)
    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    # steps_per_epoch = len(datamodule.train_dataloader())
    # print("steps_per_epoch",steps_per_epoch)
    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        # steps_per_epoch=steps_per_epoch,
        # logging=cfg.logging,
        _recursive_=False,
    )
    if cfg.model.ckpt_path is not None:
        hydra.utils.log.info(f"Loading weights from {cfg.model.ckpt_path}")
        try:
            ckpt = torch.load(cfg.model.ckpt_path)
            model.load_state_dict(ckpt["state_dict"])
            hydra.utils.log.info("Weights loaded successfully")
        except Exception as e:
            hydra.utils.log.error(f"Failed to load weights: {str(e)}")
            raise
    else:
        hydra.utils.log.info("Initializing new model")

    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logger:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logger.wandb
        os.makedirs(wandb_config.save_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            **wandb_config,
            settings=wandb.Settings(start_method="fork"),
            tags=cfg.core.tags,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logger.wandb_watch.log,
            log_freq=cfg.logger.wandb_watch.log_freq,
        )


    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)
          
    hydra.utils.log.info("Instantiating the Trainer")
    trainer = Trainer(
        default_root_dir=hydra_dir,
        logger=[wandb_logger],
        callbacks=callbacks,
        **cfg.trainer,
    )

    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    if not cfg.test_only:
        hydra.utils.log.info("Starting training!")
        trainer.fit(model, datamodule=datamodule)

    # Test
    hydra.utils.log.info("Starting testing!")
    trainer.test(model, datamodule=datamodule)



@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "config"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
