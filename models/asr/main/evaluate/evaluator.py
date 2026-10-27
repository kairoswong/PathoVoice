from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import whisper
from tqdm import tqdm

from ..arguments import parse_evaluate_args
from ..config import load_config
from ..data.dataloader import get_datasetloader
from ..data.normalizer import transcribe_normalizer, translate_normalizer
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)

METRIC_DIR = Path(__file__).resolve().parents[1] / "metric"
DEFAULT_METRICS = {"transcribe": ["cer", "emr"], "translate": ["bertscore"]}


class TaskRegistry:
    """Map task names to their handlers."""

    _registry: Dict[str, "BaseTaskHandler"] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(task_cls):
            cls._registry[name] = task_cls()
            return task_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> "BaseTaskHandler":
        if name not in cls._registry:
            raise ValueError(f"Unknown task {name!r}. Available: {list(cls._registry)}")
        return cls._registry[name]


class BaseTaskHandler:
    """Extract and normalize references for a task."""

    def normalize(self, texts: Sequence[str]) -> List[str]:
        return list(texts)

    def unpack_batch(self, batch) -> Tuple[torch.Tensor, List[str]]:
        raise NotImplementedError


@TaskRegistry.register("transcribe")
class TranscribeTask(BaseTaskHandler):
    def normalize(self, texts):
        return [transcribe_normalizer(text) for text in texts]

    def unpack_batch(self, batch):
        audio_mel, _, _, references = batch
        return audio_mel, references


@TaskRegistry.register("translate")
class TranslateTask(BaseTaskHandler):
    def normalize(self, texts):
        return [translate_normalizer(text) for text in texts]

    def unpack_batch(self, batch):
        audio_mel, _, _, _, _, _, references = batch
        return audio_mel, references


class Evaluator:
    """Decode batches and compute ASR metrics."""

    def __init__(self, model, options: whisper.DecodingOptions, metrics: Sequence[str] = None):
        import evaluate  # imported lazily: only evaluation needs it

        self.model = model
        self.options = options
        self.device = getattr(model, "device", None) or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.task = TaskRegistry.get(options.task)

        metric_names = list(metrics or DEFAULT_METRICS[options.task])
        self.metrics = {
            name: evaluate.load(str(METRIC_DIR / f"{name}.py")) for name in metric_names
        }

    @torch.no_grad()
    def run(self, dataloader) -> Tuple[pd.DataFrame, Dict[str, float]]:
        self.model.eval()
        predictions: List[str] = []
        references: List[str] = []
        total_tokens, total_time, token_speeds = 0, 0.0, []

        for batch in tqdm(dataloader, desc=self.options.task):
            audio_mel, batch_references = self.task.unpack_batch(batch)
            audio_mel = audio_mel.to(self.device)

            outputs = self.decode(audio_mel)
            predictions.extend(output.text for output in outputs)
            references.extend(batch_references)

            batch_tokens = sum(
                len(out.tokens) for out in outputs if getattr(out, "tokens", None)
            )
            if batch_tokens:
                total_tokens += batch_tokens
                token_speeds.append(batch_tokens)

            del audio_mel, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        normalized_predictions = self.task.normalize(predictions)
        normalized_references = self.task.normalize(references)
        results = self._compute_metrics(normalized_predictions, normalized_references)
        self._log_performance(total_tokens, token_speeds)

        frame = pd.DataFrame({
            "predictions": predictions,
            "references": references,
            "normalized_predictions": normalized_predictions,
            "normalized_references": normalized_references,
        })
        return frame, results

    def decode(self, audio_mel: torch.Tensor):
        """Greedy-decode a batch of log-mel spectrograms."""
        features = self.model.embed_audio(audio_mel)
        return whisper.decode(self.model, features, self.options)

    def _compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for name, metric in self.metrics.items():
            metric.add_batch(predictions=predictions, references=references)
            computed = metric.compute()
            if not isinstance(computed, dict):
                computed = {name: computed}
            for key, value in computed.items():
                output_key = key.upper() if len(self.metrics) > 1 else key
                results[output_key] = round(value * 100, 5) if isinstance(value, (int, float)) else value
        return results

    @staticmethod
    def _log_performance(total_tokens: int, token_counts: Sequence[int]):
        if not token_counts:
            return
        LOGGER.info("Decoded %d tokens in %d batches", total_tokens, len(token_counts))


def resolve_model_path(model_path: str, model_name: str) -> Optional[str]:
    """Resolve a checkpoint directory, checkpoint file, or Whisper model name."""
    candidate = Path(model_path)
    if candidate.is_dir():
        checkpoint = candidate / f"{model_name}.pt"
        return str(checkpoint) if checkpoint.is_file() else None
    if candidate.is_file():
        return str(candidate)
    if model_path in whisper.available_models():
        return model_path
    return None


def load_model(model_path: str, device: str = None):
    """Load a fine-tuned checkpoint or a stock Whisper model."""
    if model_path in whisper.available_models():
        return whisper.load_model(model_path, device=device)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return whisper.load_model(model_path, device=device)


def main(argv: Optional[list] = None):
    args = parse_evaluate_args(argv)
    config = load_config(args.config_path)

    if args.task == "translate" and not args.test_translate_file:
        raise SystemExit("--task translate requires --test-translate-file.")

    model_path = resolve_model_path(args.model_path, config.model)
    if model_path is None:
        raise FileNotFoundError(
            f"No checkpoint for '{config.model}' under {args.model_path}. Expected "
            f"'{config.model}.pt' or a checkpoint file."
        )

    device = config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
    model = load_model(model_path, device=device)
    LOGGER.info("Loaded %s on %s", model_path, device)

    test_loader = get_datasetloader(
        audio_dir=args.test_audio_dir,
        label_file=args.test_label_file,
        translate_file=args.test_translate_file if args.task == "translate" else None,
        language=config.language,
        translate_language=config.translate_language,
        batch_size=args.testing_batch_size,
        n_mels=args.n_mels,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    options = whisper.DecodingOptions(
        task=args.task,
        language=config.language if args.task == "transcribe" else config.translate_language,
        without_timestamps=True,
        fp16=args.is_fp16,
    )

    metrics = args.metrics.split(",") if args.metrics else None
    evaluator = Evaluator(model, options, metrics=metrics)
    frame, results = evaluator.run(test_loader)

    pd.set_option("display.max_rows", None)
    print(frame.to_string(), flush=True)

    if args.output_csv:
        frame.to_csv(args.output_csv, index=False)
        LOGGER.info("Predictions written to %s", args.output_csv)

    print("\nEvaluation Results:")
    for name, value in results.items():
        print(f"{name.upper()}: {value}")

    LOGGER.info("Evaluation complete.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
