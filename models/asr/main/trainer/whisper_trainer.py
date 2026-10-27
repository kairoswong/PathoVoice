from __future__ import annotations

import gc
import random
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from ..arguments import parse_train_args
from ..config import load_config
from ..data.dataloader import get_datasetloader
from ..data.spectrum import Spectrum
from ..logging_utils import get_logger, init_wandb, log_wandb

LOGGER = get_logger(__name__)

TRACKED_CONFIG_KEYS = (
    "name",
    "model",
    "language",
    "learning_rate",
    "epochs",
    "seed",
    "training_batch_size",
    "accumulation_steps",
    "validation_batch_size",
    "train_encoder",
    "train_decoder",
    "train_augment",
)


class Trainer:
    def __init__(self, config, multitask: bool = False):
        self.config = config
        self.multitask = multitask
        self.device = torch.device(config.device)
        set_seed(config.seed)

        self.accumulation_steps = int(config.accumulation_steps)
        self.theta = getattr(config, "theta", 0.5)

        Path(self.save_dir()).mkdir(parents=True, exist_ok=True)
        self.train_loader, self.val_loader = self._init_dataloaders()
        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

    def save_dir(self) -> str:
        """Checkpoint directory derived from the config unless pinned explicitly."""
        key = "multitask_save_dir" if self.multitask else "save_dir"
        explicit = getattr(self.config, key, None)
        if explicit:
            return explicit

        task = "multitask" if self.multitask else "transcribe"
        augment = "_aug" if getattr(self.config, "train_augment", False) else ""
        return (
            f"{self.config.checkpoint_dir}/"
            f"{task}{augment}/seed{self.config.seed}/{self.config.model}"
        )

    def run(self):
        self.epoch_loop()

    def _init_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError

    def _init_model(self):
        raise NotImplementedError

    def _init_optimizer(self):
        raise NotImplementedError

    def _init_scheduler(self):
        steps_per_epoch = len(self.train_loader) // self.accumulation_steps
        total_steps = max(self.config.epochs * steps_per_epoch, 1)
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def cross_entropy_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.transpose(1, 2), target)

    @staticmethod
    def cleanup_gpu(*tensors):
        for tensor in tensors:
            del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def save_model(self, save_path: str):
        state_dict = {k: v.half() for k, v in self.model.state_dict().items()}
        dims = asdict(self.model.dims) if hasattr(self.model, "dims") else None
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": state_dict, "dims": dims}, save_path)
        LOGGER.info("Model saved to %s", save_path)

    @torch.no_grad()
    def evaluate_model(self) -> float:
        raise NotImplementedError

    def train_single_epoch(self) -> float:
        raise NotImplementedError

    def epoch_loop(self):
        """Train for ``config.epochs`` epochs, validating every ``val_epochs``."""
        min_loss = self.evaluate_model()
        LOGGER.info("Initial validation loss: %.6f", min_loss)
        log_wandb({"Validation_Loss": min_loss})

        for epoch in tqdm(range(1, self.config.epochs + 1), desc="epochs"):
            epoch_loss = self.train_single_epoch()
            LOGGER.info("Epoch %d training loss: %.8f", epoch, epoch_loss)
            log_wandb({"Epoch_Loss": epoch_loss})

            if self.config.save_checkpoints:
                self._save_epoch_checkpoint(epoch)

            if epoch % self.config.val_epochs != 0:
                continue

            val_loss = self.evaluate_model()
            LOGGER.info("Epoch %d validation loss: %.6f", epoch, val_loss)
            log_wandb({"Validation_Loss": val_loss})

            if val_loss < min_loss:
                min_loss = val_loss
                self.save_model(f"{self.save_dir()}/{self.config.model}.pt")

            self.save_model(f"{self.save_dir()}/last_model.pt")
            self.cleanup_gpu()

    def _save_epoch_checkpoint(self, epoch: int):
        checkpoint_dir = Path(self.save_dir()) / f"checkpoint_{epoch}"
        try:
            self.save_model(str(checkpoint_dir / f"{self.config.model}.pt"))
        except Exception as exc:  # keep training even if a checkpoint fails to write
            LOGGER.warning("Failed to save model checkpoint: %s", exc)


class WhisperTrainer(Trainer):
    def _init_model(self):
        LOGGER.info("Loading Whisper '%s' on %s", self.config.model, self.device)
        return whisper.load_model(self.config.model, self.device)

    def _init_dataloaders(self):
        train_loader = get_datasetloader(
            audio_dir=self.config.train_audio_dir,
            label_file=self.config.train_label_file,
            translate_file=self.config.train_translate_file if self.multitask else None,
            language=self.config.language,
            translate_language=self.config.translate_language,
            transform=Spectrum(
                augment=getattr(self.config, "train_augment", False),
            ),
            batch_size=self.config.training_batch_size,
            n_mels=self.config.mel_channels,
            seed=self.config.seed,
            shuffle=True,
            num_workers=getattr(self.config, "num_workers", 2),
            pin_memory=getattr(self.config, "pin_memory", True),
        )
        val_loader = get_datasetloader(
            audio_dir=self.config.val_audio_dir,
            label_file=self.config.val_label_file,
            translate_file=self.config.val_translate_file if self.multitask else None,
            language=self.config.language,
            translate_language=self.config.translate_language,
            batch_size=self.config.validation_batch_size,
            n_mels=self.config.mel_channels,
            seed=self.config.seed,
            shuffle=False,
            num_workers=getattr(self.config, "num_workers", 2),
            pin_memory=getattr(self.config, "pin_memory", True),
        )
        LOGGER.info(
            "Dataset sizes: %d training samples, %d validation samples",
            len(train_loader.dataset),
            len(val_loader.dataset),
        )
        return train_loader, val_loader

    def _init_optimizer(self):
        self._configure_gradients()

        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        return torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta_1, self.config.beta_2),
            eps=self.config.epsilon,
        )

    def _configure_gradients(self):
        """Freeze encoder and/or decoder according to the config."""
        train_encoder = bool(getattr(self.config, "train_encoder", True))
        train_decoder = bool(getattr(self.config, "train_decoder", False))
        for name, param in self.model.named_parameters():
            if name.startswith("encoder"):
                param.requires_grad = train_encoder
            elif name.startswith("decoder"):
                param.requires_grad = train_decoder
            else:
                param.requires_grad = True

        for label, trained in (("encoder", train_encoder), ("decoder", train_decoder)):
            LOGGER.info("%s: %s", label, "training" if trained else "frozen")

    def get_logits(self, input_data: torch.Tensor, decoder_in: torch.Tensor) -> torch.Tensor:
        features = self.model.embed_audio(input_data)
        return self.model.logits(decoder_in, audio_features=features)

    def get_loss_for_task(
        self,
        input_data: torch.Tensor,
        decoder_in: torch.Tensor,
        decoder_out: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.get_logits(input_data, decoder_in)
        return self.cross_entropy_loss(logits, decoder_out)

    def _batch_loss(self, batch) -> torch.Tensor:
        """Combined loss of one batch (multitask mixes transcribe and translate)."""
        if self.multitask:
            (audio_mel, transcript_in, transcript_out, _, translation_in, translation_out, _) = batch
            input_data = audio_mel.to(self.device)
            loss_transcribe = self.get_loss_for_task(
                input_data,
                transcript_in.to(self.device),
                transcript_out.to(self.device),
            )
            loss_translate = self.get_loss_for_task(
                input_data,
                translation_in.to(self.device),
                translation_out.to(self.device),
            )
            return self.theta * loss_transcribe + (1 - self.theta) * loss_translate

        audio_mel, decoder_in, decoder_out, _ = batch
        return self.get_loss_for_task(
            audio_mel.to(self.device),
            decoder_in.to(self.device),
            decoder_out.to(self.device),
        )

    @torch.no_grad()
    def evaluate_model(self) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in tqdm(self.val_loader, desc="valid", leave=False):
            loss = self._batch_loss(batch)
            total_loss += loss.item()
            self.cleanup_gpu(loss)

        self.model.train()
        return total_loss / max(len(self.val_loader), 1)

    def train_single_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        last_step = len(self.train_loader) - 1

        for step, batch in enumerate(tqdm(self.train_loader, desc="train", leave=False)):
            loss = self._batch_loss(batch) / self.accumulation_steps
            loss.backward()

            if (step + 1) % self.accumulation_steps == 0 or step == last_step:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            log_wandb({"Training_Loss": loss.item() * self.accumulation_steps})

        return running_loss * self.accumulation_steps / max(len(self.train_loader), 1)


def get_trainer(config, multitask: bool = False) -> Trainer:
    """Instantiate the trainer matching ``config.name``."""
    if config.name == "main":
        return WhisperTrainer(config, multitask=multitask)
    raise ValueError(f"Unsupported config.name: {config.name!r} (expected 'main').")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main(argv: Optional[list] = None):
    args = parse_train_args(argv)
    config = load_config(args.config_path)

    if args.multitask:
        missing = [
            key for key in ("train_translate_file", "val_translate_file")
            if not getattr(config, key, None) or not Path(getattr(config, key)).is_file()
        ]
        if missing:
            raise SystemExit(
                f"--multitask requires existing {', '.join(missing)} per {config.config_path}."
            )

    init_wandb(
        project=config.name,
        run_name=f"{config.name}_{config.model}",
        config={key: getattr(config, key, None) for key in TRACKED_CONFIG_KEYS},
        extra={"multitask": args.multitask},
    )

    try:
        trainer = get_trainer(config, multitask=args.multitask)
        trainer.run()
    finally:
        from ..logging_utils import finish_wandb
        finish_wandb()
    return trainer


if __name__ == "__main__":
    main()
