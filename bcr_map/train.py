"""MIL training (LightningCLI, k-fold)."""

import logging
import os
import uuid
import torch
import torch.backends.cuda
import torch.backends.cudnn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from utils.mil_data import MILDataModule
from utils.mil_model import MILClassificationModel

logger = logging.getLogger(__name__)

# Default config when --config is omitted (HiPT + ViT-MIL; override with --config path).
_DEFAULT_MIL_CONFIG = "configs/mil_training.yaml"


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def get_version_name(config):
    """Version string for checkpoint and log layout.

    Default: base version from config plus a short UUID (avoids collisions).
    Set BCR_MAP_STABLE_VERSION=1 to use the config version exactly (reproducible paths).
    """
    base_version = config.get("version", "") or "run"
    if _env_truthy("BCR_MAP_STABLE_VERSION"):
        return base_version
    unique_id = str(uuid.uuid4())[:4]
    return f"{base_version}_{unique_id}"


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.add_argument("--k_folds", type=int, default=3)
        parser.add_argument("--run_test", type=bool, default=False)
        parser.add_argument("--version", type=str, default="")
        parser.set_defaults(
            {
                "trainer.logger": [
                    {"class_path": "pytorch_lightning.loggers.CSVLogger", "init_args": {"save_dir": "logs"}},
                    {"class_path": "pytorch_lightning.loggers.WandbLogger", "init_args": {"project": "BCR_map_paper_analysis"}},
                ],
                "model_checkpoint.monitor": "val_acc",
                "model_checkpoint.mode": "max",
                "model_checkpoint.filename": "fold-{fold}-step-{step}-{val_acc:.4f}",
                "model_checkpoint.save_last": True,
            }
        )

        parser.link_arguments("data.num_classes", "model.n_classes", apply_on="instantiate")

    def before_instantiate_classes(self) -> None:
        """Runs before instantiating the classes.

        (We remove the part that attaches a UUID here so that it happens only once.)
        """
        # -- Remove the UUID logic from here --
        # Instead, we now assume `self.config['version']` is already set (including UUID).
        # So we only do the standard "fold_x" dirpath and logger name updates.

        # Ensure dirpath for the checkpoint is set properly
        if "version" not in self.config:
            self.config["version"] = ""  # fallback

        # Optional: CSV only (no Weights & Biases) for sharing / offline runs
        if _env_truthy("BCR_MAP_CSV_LOGGER_ONLY"):
            self.config["trainer.logger"] = [
                {
                    "class_path": "pytorch_lightning.loggers.CSVLogger",
                    "init_args": {
                        "save_dir": "logs",
                        "version": self.config["version"],
                    },
                }
            ]
        elif "trainer.logger" not in self.config or not self.config["trainer.logger"]:
            self.config["trainer.logger"] = [
                {"class_path": "pytorch_lightning.loggers.CSVLogger", "init_args": {"save_dir": "logs"}},
                {"class_path": "pytorch_lightning.loggers.WandbLogger", "init_args": {"project": "BCR_map_paper_analysis"}},
            ]

        # You can still dynamically set checkpoint path and logger to use that version
        self.config["model_checkpoint.dirpath"] = os.path.join(
            "checkpoints",
            self.config["version"],
            f"fold_{self.config['data.fold']}"
        )

        loggers = self.config.get("trainer.logger") or []
        if loggers:
            if "init_args" not in loggers[0]:
                loggers[0]["init_args"] = {}
            loggers[0]["init_args"]["save_dir"] = "logs"
            loggers[0]["init_args"]["version"] = self.config["version"]
        if len(loggers) >= 2:
            if "init_args" not in loggers[1]:
                loggers[1]["init_args"] = {}
            loggers[1]["init_args"]["name"] = self.config["version"]
            loggers[1]["init_args"].setdefault("project", "BCR_map_paper_analysis")


def _inject_default_config_argv():
    """If no --config is passed, default to MIL yaml (HiPT default is inside that file)."""
    import sys

    if "--help" in sys.argv or "-h" in sys.argv:
        return
    if any(a == "--config" or a.startswith("--config=") for a in sys.argv):
        return
    # LightningCLI expects: python main.py --config path ...
    sys.argv[1:1] = ["--config", _DEFAULT_MIL_CONFIG]


def cli_main() -> None:
    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    _inject_default_config_argv()
    logger.info("config: %s", _DEFAULT_MIL_CONFIG)

    base_cli = MyLightningCLI(
        MILClassificationModel,
        MILDataModule,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={"check_val_every_n_epoch": None},
        run=False,
    )

    ##################################################
    # 3) Attach UUID to version **only once** here
    ##################################################
    # If the user didn't pass a --version from CLI, we create one

    base_cli.config["version"] = get_version_name(base_cli.config)

    # Now base_cli.config["version"] has the unique version we want to reuse.
    k_folds = base_cli.config["k_folds"]
    run_test = base_cli.config["run_test"]

    ##################################################
    # 4) Run K-Fold Cross Validation
    ##################################################
    for fold in range(k_folds):
        logger.info("%s\nRunning fold %s\n%s", "=" * 40, fold, "=" * 40)

        # Update the fold number in config
        base_cli.config["data.fold"] = fold

        # Instantiate a new CLI for each fold, reusing the same version string
        cli = MyLightningCLI(
            MILClassificationModel,
            MILDataModule,
            args=base_cli.config,
            save_config_kwargs={"overwrite": True},
            trainer_defaults={"check_val_every_n_epoch": None},
            run=False,
        )

        # 5) Train the model
        cli.trainer.fit(cli.model, cli.datamodule)

        # 6) Optionally run tests
        if run_test:
            cli.trainer.test(cli.model, cli.datamodule)

        # 7) Save config in checkpoint dir with simpler error handling
        try:
            if (hasattr(cli.trainer, 'logger') and
                cli.trainer.logger is not None and
                hasattr(cli.trainer.logger, 'save_dir') and
                cli.trainer.logger.save_dir is not None):

                config_src = os.path.join(cli.trainer.logger.save_dir, "config.yaml")

                # Try to find checkpoint directory from version and fold
                expected_checkpoint_dir = os.path.join("checkpoints", base_cli.config["version"], f"fold_{fold}")
                if os.path.exists(expected_checkpoint_dir):
                    config_dst = os.path.join(expected_checkpoint_dir, "config.yaml")
                    if os.path.exists(config_src):
                        os.rename(config_src, config_dst)
        except (OSError, AttributeError) as e:
            logger.warning("Could not move config file: %s", e)
