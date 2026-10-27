from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Union

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
PACKAGE_ROOT = CONFIG_DIR.parent


def resolve_config_path(config: Union[str, Path]) -> Path:
    """Resolve a config given as a path, a file name, or a bare config stem."""
    candidate = Path(config)
    if candidate.is_file():
        return candidate

    for name in (f"{candidate.name}.yaml", f"{candidate.name}.yml"):
        path = CONFIG_DIR / name
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Config '{config}' not found. Pass an existing YAML path or the name of a "
        f"file in {CONFIG_DIR}."
    )


def load_config(config: Union[str, Path]) -> SimpleNamespace:
    """Read a YAML config and return it as an attribute-accessible namespace."""
    path = resolve_config_path(config)
    with open(path, "r", encoding="utf-8") as fp:
        data: Dict[str, Any] = yaml.safe_load(fp) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping, got {type(data).__name__}.")

    data["config_path"] = str(path)
    return SimpleNamespace(**_normalize(data, path))


def _resolve_data_path(value: Any) -> Any:
    if value is None or not str(value).strip():
        return value
    candidate = Path(str(value))
    if candidate.is_absolute():
        return str(candidate)
    return str(PACKAGE_ROOT / candidate)


def _normalize(data: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """Fill in defaults and normalize legacy configuration keys."""
    data = dict(data)

    # Migrate the legacy decoder-freeze option.
    legacy = data.pop("train_only_encoder", None)
    if legacy is not None and "train_encoder" not in data and "train_decoder" not in data:
        warnings.warn(
            f"{path.name}: 'train_only_encoder' is deprecated, use "
            f"'train_encoder'/'train_decoder' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        data["train_encoder"] = True
        data["train_decoder"] = not bool(legacy)
    elif legacy is not None:
        warnings.warn(
            f"{path.name}: ignoring 'train_only_encoder' because "
            f"'train_encoder'/'train_decoder' are set.",
            DeprecationWarning,
            stacklevel=2,
        )

    data.setdefault("name", "main")
    data.setdefault("device", "cuda")
    data.setdefault("train_encoder", True)
    data.setdefault("train_decoder", False)
    data.setdefault("train_augment", False)
    data.setdefault("save_checkpoints", False)
    data.setdefault("seed", 3407)
    data.setdefault("epochs", 10)
    data.setdefault("warmup_ratio", 0.2)
    data.setdefault("val_epochs", 1)
    data.setdefault("accumulation_steps", 1)
    data.setdefault("max_grad_norm", 1.0)
    data.setdefault("beta_1", 0.9)
    data.setdefault("beta_2", 0.99)
    data.setdefault("weight_decay", 1.0e-2)
    data.setdefault("epsilon", 1.0e-8)
    data.setdefault("theta", 0.5)
    data.setdefault("mel_channels", 80)
    data.setdefault("language", "zh")
    data.setdefault("translate_language", "en")
    data.setdefault("num_workers", 2)
    data.setdefault("pin_memory", True)
    # Resolve relative paths from the ASR package root.
    data.setdefault("checkpoint_dir", "checkpoints")
    data.setdefault("train_audio_dir", "datasets/train_set/wav")
    data.setdefault("train_label_file", "datasets/train_set/transcript")
    data.setdefault("val_audio_dir", "datasets/val_set/wav")
    data.setdefault("val_label_file", "datasets/val_set/transcript")
    data.setdefault("train_translate_file", "datasets/train_set/translate")
    data.setdefault("val_translate_file", "datasets/val_set/translate")

    for key in (
        "checkpoint_dir",
        "train_audio_dir", "train_label_file", "train_translate_file",
        "val_audio_dir", "val_label_file", "val_translate_file",
    ):
        data[key] = _resolve_data_path(data[key])

    for key in ("model", "learning_rate", "training_batch_size", "validation_batch_size"):
        if key not in data:
            raise KeyError(f"Config {path} is missing required key '{key}'.")

    return data
