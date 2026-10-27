from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, TypeVar

import numpy as np
import torch
from torch import Tensor
from whisper.normalizers.english import EnglishTextNormalizer

T = TypeVar("T")


class TextNormalizer(ABC):
    """Common behaviour for text normalizers (singleton, self-test helper)."""

    _instance: Optional["TextNormalizer"] = None

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize ``text`` according to the concrete normalizer's rules."""
        raise NotImplementedError

    @classmethod
    def get_instance(cls: Type[T]) -> T:
        """Return the process-wide singleton instance of ``cls``."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_test_cases(cls) -> Dict[str, str]:
        """Return ``{input: expected}`` pairs used by :meth:`run_tests`."""
        return {}

    @classmethod
    def run_tests(cls) -> bool:
        """Run the normalizer against :meth:`get_test_cases` and report failures."""
        instance = cls.get_instance()
        print(f"Running tests for {cls.__name__}...")

        all_passed = True
        for i, (input_text, expected) in enumerate(cls.get_test_cases().items(), 1):
            result = instance.normalize(input_text)
            if result != expected:
                print(f"Test case {i} failed:")
                print(f"Input:    '{input_text}'")
                print(f"Expected: '{expected}'")
                print(f"Got:      '{result}'")
                all_passed = False

        if all_passed:
            print("All tests passed!")

        return all_passed


class TranscribeNormalizer(TextNormalizer):
    """Normalizer for transcribed Chinese text."""

    CHINESE_CHARS = r"\u4e00-\u9fff"

    def __init__(self):
        self._t2s = None  # resolved lazily so zhconv stays optional at import time

    def _ensure_t2s(self):
        if self._t2s is None:
            from zhconv import convert
            self._t2s = lambda text: convert(text, "zh-cn")

    def normalize(self, text: str) -> str:
        """Convert to simplified Chinese and drop stutter repetitions."""
        if not text:
            return ""

        self._ensure_t2s()
        text = self._t2s(text)

        text = re.sub(f"[^{self.CHINESE_CHARS}\\s]", "", text)

        # Reduce runs of the same character to at most two occurrences.
        text = re.sub(f"([{self.CHINESE_CHARS}])\\1{{2,}}", lambda m: m.group(1) * 2, text)

        pattern = re.compile(f"([{self.CHINESE_CHARS}]{{2,}})(?:\\1)+")
        previous = None
        while previous != text:
            previous = text
            text = pattern.sub(r"\1", text)

        words = [w for w in text.split() if w]
        normalized = []
        for word in words:
            if normalized and normalized[-1] == word:
                continue
            normalized.append(word)
        text = " ".join(normalized)

        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def get_test_cases() -> Dict[str, str]:
        return {
            "": "",
            "你好你好你好": "你好",
            "你你你好": "你你好",
            "中文 中文 文字": "中文 文字",
            "测试测试 测试测试": "测试",
            "妳好": "你好",
            "繁體中文": "繁体中文",
        }


class TranslateNormalizer(TextNormalizer):
    """Normalizer for English translations."""

    def __init__(self):
        super().__init__()
        self.whisper_normalizer = EnglishTextNormalizer()

    def normalize(self, text: str) -> str:
        """Apply the official Whisper English text normalizer."""
        if not text:
            return ""
        return self.whisper_normalizer(text)

    @staticmethod
    def get_test_cases() -> Dict[str, str]:
        return {
            "": "",
            "two and a half": "2.5",
            "I'm happy": "i am happy",
            "hello  world": "hello world",
            "They won't go": "they will not go",
            "twenty five dollars": "25 dollars",
            "Uh (noise) this is weird": "this is weird",
        }


class AudioNormalizer:
    """Waveform-level normalisation helpers."""

    @staticmethod
    def rms_normalize(x: Tensor, target_level: float = 0.05, eps: float = 1e-8) -> Tensor:
        """Scale the signal so that its RMS equals ``target_level``."""
        if isinstance(x, torch.Tensor):
            rms = torch.sqrt(torch.mean(x ** 2) + eps)
            return target_level * x / rms
        if isinstance(x, np.ndarray):
            rms = np.sqrt(np.mean(x ** 2) + eps)
            return target_level * x / rms
        raise TypeError(f"Unsupported input type: {type(x)}. Must be torch.Tensor or np.ndarray.")


def rms_normalize(x: Tensor, target_level: float = 0.05, eps: float = 1e-8) -> Tensor:
    """Normalize waveform amplitude to a target RMS level."""
    return AudioNormalizer.rms_normalize(x, target_level, eps)


def transcribe_normalizer(text: str) -> str:
    """Normalize transcribed text."""
    return TranscribeNormalizer.get_instance().normalize(text)


def translate_normalizer(text: str) -> str:
    """Normalize translated text."""
    return TranslateNormalizer.get_instance().normalize(text)
