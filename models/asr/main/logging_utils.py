from __future__ import annotations

import logging
from typing import Any, Dict

try:
    import wandb
except ImportError:
    wandb = None

_LOGGERS: Dict[str, logging.Logger] = {}
WANDB_ACTIVE = False


def get_logger(name: str = "main") -> logging.Logger:
    """Return a configured logger; repeated calls return the same object."""
    logger = _LOGGERS.get(name)
    if logger is not None:
        return logger

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    _LOGGERS[name] = logger
    return logger


def wandb_available() -> bool:
    return wandb is not None


def init_wandb(
    project: str,
    run_name: str,
    config: Dict[str, Any] = None,
    entity: str = None,
    extra: Dict[str, Any] = None,
) -> bool:
    """Start a W&B run if available. Returns ``True`` when tracking is active."""
    global WANDB_ACTIVE
    if wandb is None:
        WANDB_ACTIVE = False
        return False

    payload = dict(config or {})
    payload.update(extra or {})
    try:
        wandb.init(
            entity=entity,
            project=project,
            name=run_name,
            config=payload,
            settings=wandb.Settings(init_timeout=30),
        )
        WANDB_ACTIVE = True
    except Exception as exc:
        get_logger().warning("wandb.init failed, continuing without tracking: %s", exc)
        WANDB_ACTIVE = False
    return WANDB_ACTIVE


def log_wandb(metrics: Dict[str, Any]):
    """Log metrics to W&B when a run is active; otherwise do nothing."""
    if WANDB_ACTIVE and wandb is not None:
        wandb.log(metrics)


def finish_wandb():
    """Close the active W&B run, if any."""
    global WANDB_ACTIVE
    if WANDB_ACTIVE and wandb is not None:
        wandb.finish()
    WANDB_ACTIVE = False
