from __future__ import annotations

import argparse
from typing import Iterable, List, Sequence, Tuple

TRAINING_ARGS: List[Tuple[Sequence[str], dict]] = [
    [("--config-path",), dict(
        type=str,
        required=True,
        help="Path to the training config YAML (e.g. models/asr/configs/tiny.yaml)",
    )],
    [("--multitask",), dict(
        action="store_true",
        help="Enable multitask training (transcribe + translate); the config must define "
             "'train_translate_file' and 'val_translate_file'",
    )],
]

EVALUATE_ARGS: List[Tuple[Sequence[str], dict]] = [
    [("--config-path",), dict(
        type=str,
        required=True,
        help="Path to the training config YAML that defines the model size",
    )],
    [("--model-path",), dict(
        type=str,
        required=True,
        help="Directory holding the fine-tuned '<model>.pt', or the .pt file itself",
    )],
    [("--test-audio-dir",), dict(
        type=str,
        required=True,
        help="Directory that holds the test audio files",
    )],
    [("--test-label-file",), dict(
        type=str,
        required=True,
        help="Label file with reference transcripts ('<id> <text>' per line), or a "
             "directory whose *.txt label files are merged (e.g. transcript/)",
    )],
    [("--test-translate-file",), dict(
        type=str,
        default=None,
        help="Label file with the reference translations, or a directory of label "
             "files, required for --task translate",
    )],
    [("--task",), dict(
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task to evaluate",
    )],
    [("--metrics",), dict(
        type=str,
        default=None,
        help="Comma separated metric names, overriding the task defaults "
             "(transcribe: cer,emr; translate: bertscore)",
    )],
    [("--testing-batch-size",), dict(
        type=int,
        default=10,
        help="Number of samples in each batch during testing",
    )],
    [("--n-mels",), dict(
        type=int,
        default=80,
        help="Number of mel channels",
    )],
    [("--num-workers",), dict(
        type=int,
        default=0,
        help="DataLoader worker processes used during evaluation",
    )],
    [("--is-fp16",), dict(
        action="store_true",
        help="Use fp16 precision for evaluation",
    )],
    [("--output-csv",), dict(
        type=str,
        default=None,
        help="Optional path for a CSV dump of predictions and references",
    )],
]


def add_arguments(parser: argparse.ArgumentParser, arguments: Iterable[Tuple[Sequence[str], dict]]):
    """Add argument definitions to a parser."""
    for flags, kwargs in arguments:
        parser.add_argument(*flags, **kwargs)
    return parser


def build_training_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune the model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return add_arguments(parser, TRAINING_ARGS)


def build_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return add_arguments(parser, EVALUATE_ARGS)


def parse_train_args(argv: Sequence[str] = None) -> argparse.Namespace:
    return build_training_parser().parse_args(argv)


def parse_evaluate_args(argv: Sequence[str] = None) -> argparse.Namespace:
    return build_evaluate_parser().parse_args(argv)
