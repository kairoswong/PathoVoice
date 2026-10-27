from __future__ import annotations

import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from whisper.tokenizer import get_tokenizer

from .spectrum import Spectrum

AUDIO_EXTENSIONS = (
    ".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".wma",
    ".webm", ".mp4", ".mpeg", ".mpga", ".aiff", ".aif", ".caf", ".amr", ".3gp",
)


def _is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def load_label_file(label_file: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    files = _label_files(label_file)
    for path in files:
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(maxsplit=1)
                labels[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
    return labels


def _label_files(label_file: str) -> List[Path]:
    """Resolve a label-file argument to files."""
    path = Path(label_file)
    if path.is_file():
        return [path]
    if path.is_dir():
        found = sorted(p for p in path.rglob("*.txt") if p.is_file())
        if found:
            return found
        raise FileNotFoundError(
            f"Label directory {label_file} contains no .txt label files."
        )
    raise FileNotFoundError(f"Label file {label_file} not found.")


def _strip_audio_extension(name: str) -> str:
    """Remove a trailing audio extension and URL fragment from an id."""
    for separator in ("?", "#"):
        index = name.rfind(separator)
        if index != -1:  # tolerate ids copied from URLs
            name = name[:index]
    suffix = Path(name).suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        return name[: -len(suffix)]
    return name


class CustomDataset(Dataset):
    def __init__(
        self,
        audio_dir: str = None,
        label_file: str = None,
        translate_file: str = None,
        language: str = "zh",
        translate_language: str = "en",
        transform: Spectrum = None,
        max_tokens_context: int = 448,
        n_mels: int = 80,
        tokenizer_type: str = "multilingual",
    ) -> None:
        if label_file is None:
            raise ValueError("'label_file' is required.")
        if transform is None:
            transform = Spectrum()

        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.n_mels = n_mels
        self.max_tokens_context = max_tokens_context
        self.language = language
        self.translate_language = translate_language
        self.spectrum_transform = transform
        self.tokenizer_type = tokenizer_type
        self.tokenizer = get_tokenizer(multilingual=(tokenizer_type == "multilingual"))

        self.transcripts = load_label_file(label_file)
        translations = load_label_file(translate_file) if translate_file else None
        self.translations = (
            {self._record_key(key): value for key, value in translations.items()}
            if translations is not None
            else None
        )
        if not self.transcripts:
            raise ValueError(f"Label file {label_file} contains no usable lines.")

        self.audio_index: Dict[str, Path] = {}
        if self.audio_dir is not None:
            if not self.audio_dir.is_dir():
                raise FileNotFoundError(f"Audio directory {self.audio_dir} not found.")
            self.audio_index = self._build_audio_index()

        self.records = self._build_records()

    def _build_audio_index(self) -> Dict[str, Path]:
        wanted = {self._record_key(name) for name in self.transcripts}
        index: Dict[str, Path] = {}

        for path in sorted(self.audio_dir.rglob("*")):
            if not _is_audio_file(path):
                continue
            key = path.stem
            if key in wanted:
                index.setdefault(key, path)

        return index

    def _build_records(self) -> List[Tuple[Path, str, Optional[str]]]:
        records: List[Tuple[Path, str, Optional[str]]] = []
        missing: List[str] = []

        for raw_id, transcript in self.transcripts.items():
            key = self._record_key(raw_id)
            if self.translations is not None and key not in self.translations:
                raise KeyError(
                    f"'{raw_id}' has a transcript but no translation; every sample needs "
                    f"both when a translate file is configured."
                )

            audio_path = None
            if self.audio_dir is not None:
                audio_path = self.audio_index.get(key)
                if audio_path is None:
                    missing.append(raw_id)
                    continue

            records.append((
                audio_path,
                transcript,
                self.translations[key] if self.translations is not None else None,
            ))

        if missing:
            preview = ", ".join(missing[:5])
            more = f" ... (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            raise FileNotFoundError(
                f"{len(missing)} label entr{'y' if len(missing) == 1 else 'ies'} have no matching "
                f"audio file under {self.audio_dir}: {preview}{more}"
            )
        if not records:
            raise ValueError("Dataset is empty after matching labels with audio files.")
        return records

    def _record_key(self, name: str) -> str:
        return _strip_audio_extension(name.strip())

    def __len__(self) -> int:
        return len(self.records)

    def _get_text_tokens(self, text: str) -> List[int]:
        return self.tokenizer.encode(text.strip())

    def _get_special_tokens(self, task: str, is_empty: bool, language: str) -> List[int]:
        if is_empty:
            special_tokens = [self.tokenizer.sot, self.tokenizer.no_speech]
        elif task == "transcribe":
            special_tokens = [
                self.tokenizer.sot,
                self.tokenizer.special_tokens[f"<|{language}|>"],
                self.tokenizer.special_tokens["<|transcribe|>"],
            ]
        elif task == "translate":
            special_tokens = [
                self.tokenizer.sot,
                self.tokenizer.special_tokens[f"<|{language}|>"],
                self.tokenizer.special_tokens["<|translate|>"],
            ]
        else:
            raise ValueError(f"Invalid task: {task}. Must be 'transcribe' or 'translate'.")

        special_tokens.append(self.tokenizer.no_timestamps)
        return special_tokens

    def _decoder_output(self, special_tokens: Sequence[int], text_tokens: Sequence[int]) -> List[int]:
        return list(special_tokens[1:]) + list(text_tokens) + [self.tokenizer.eot]

    def _decoder_tokens(self, text: str, task: str, language: str) -> Tuple[torch.Tensor, torch.Tensor]:
        text_tokens = self._get_text_tokens(text)
        special_tokens = self._get_special_tokens(
            task=task,
            is_empty=len(text_tokens) == 0,
            language=language,
        )

        decoder_input = special_tokens + text_tokens
        if len(decoder_input) > self.max_tokens_context:
            raise ValueError(
                f"Input is too long for the {self.max_tokens_context} token context: "
                f"length {len(decoder_input)}."
            )
        decoder_output = self._decoder_output(special_tokens, text_tokens)
        return (
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(decoder_output, dtype=torch.long),
        )

    def __getitem__(self, index: int):
        audio_path, transcript, translation = self.records[index]

        audio_mel = None
        if audio_path is not None:
            augment_params = (
                self.spectrum_transform.get_time_augment_params()
                if self.spectrum_transform.augment
                else None
            )
            audio_mel = self.spectrum_transform.audio_mel(
                audio_path,
                n_mels=self.n_mels,
                augment_params=augment_params,
            )
        else:
            audio_mel = torch.empty(0)

        decoder_in, decoder_out = self._decoder_tokens(
            transcript, task="transcribe", language=self.language
        )

        if translation is not None:
            translation_in, translation_out = self._decoder_tokens(
                translation, task="translate", language=self.translate_language
            )
            return (
                audio_mel,
                decoder_in,
                decoder_out,
                transcript,
                translation_in,
                translation_out,
                translation,
            )

        return audio_mel, decoder_in, decoder_out, transcript


def _pad_mels(mels: Sequence[torch.Tensor], padding_value: float = 0.0) -> torch.Tensor:
    """Stack spectrograms after right-padding the time axis."""
    max_frames = max(mel.shape[-1] for mel in mels)
    padded = [
        torch.nn.functional.pad(mel, (0, max_frames - mel.shape[-1]), value=padding_value)
        for mel in mels
    ]
    return torch.stack(padded, dim=0)


def collate_transcribe(batch):
    audio_mel, decoder_in, decoder_out, transcripts = zip(*batch)
    return (
        _pad_mels(audio_mel),
        pad_sequence(decoder_in, batch_first=True, padding_value=0),
        pad_sequence(decoder_out, batch_first=True, padding_value=-100),
        list(transcripts),
    )


def collate_multitask(batch):
    (
        audio_mel,
        decoder_transcript_input,
        decoder_transcript_output,
        transcripts,
        decoder_translation_input,
        decoder_translation_output,
        translations,
    ) = zip(*batch)
    return (
        _pad_mels(audio_mel),
        pad_sequence(decoder_transcript_input, batch_first=True, padding_value=0),
        pad_sequence(decoder_transcript_output, batch_first=True, padding_value=-100),
        list(transcripts),
        pad_sequence(decoder_translation_input, batch_first=True, padding_value=0),
        pad_sequence(decoder_translation_output, batch_first=True, padding_value=-100),
        list(translations),
    )


def _worker_init_fn(worker_id: int, seed: int):
    """Seed each data-loader worker."""
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_datasetloader(
    audio_dir: str = None,
    label_file: str = None,
    translate_file: str = None,
    language: str = "zh",
    translate_language: str = "en",
    seed: int = 3407,
    transform: Spectrum = None,
    batch_size: int = 16,
    n_mels: int = 80,
    shuffle: bool = True,
    persistent_workers: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    """Build a data loader for training, validation, or evaluation."""
    if transform is None:
        transform = Spectrum()

    dataset = CustomDataset(
        audio_dir=audio_dir,
        label_file=label_file,
        translate_file=translate_file,
        language=language,
        translate_language=translate_language,
        transform=transform,
        n_mels=n_mels,
    )

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_multitask if translate_file else collate_transcribe,
        worker_init_fn=partial(_worker_init_fn, seed=seed),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
    return DataLoader(**loader_kwargs)
