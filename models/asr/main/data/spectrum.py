from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import whisper
from whisper.audio import N_FRAMES

from .normalizer import rms_normalize


class Spectrum:
    def __init__(self, augment: bool = False):
        self.augment = augment

    def get_time_augment_params(self) -> dict:
        return {
            "stretch_factor": float(np.random.uniform(0.9, 1.1)),
            "shift_ratio": float(np.random.uniform(-0.05, 0.05)),
            "drop_prob": bool(np.random.rand() < 0.3),
        }

    def apply_specaugment(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.clone()
        n_mels, n_frames = mel.shape

        for _ in range(2):
            f = int(np.random.uniform(0, min(12, n_mels // 2)))
            f0 = int(np.random.uniform(0, max(1, n_mels - f)))
            mel[f0:f0 + f, :] = 0.0

        for _ in range(2):
            t = int(np.random.uniform(0, min(50, n_frames // 3)))
            t0 = int(np.random.uniform(0, max(1, n_frames - t)))
            mel[:, t0:t0 + t] = 0.0

        return mel

    @staticmethod
    def _apply_time_augment(x: torch.Tensor, params: dict) -> torch.Tensor:
        stretch_factor = params["stretch_factor"]
        if abs(stretch_factor - 1.0) > 1e-3:
            orig_len = x.shape[-1]
            new_len = max(int(orig_len * stretch_factor), 4)
            x = torch.nn.functional.interpolate(
                x.view(1, 1, -1), size=new_len, mode="linear", align_corners=False
            ).view(-1)

        shift_len = int(x.shape[-1] * params["shift_ratio"])
        if shift_len > 0:
            x = torch.cat([x[shift_len:], torch.zeros(shift_len, dtype=x.dtype)])
        elif shift_len < 0:
            x = torch.cat([torch.zeros(-shift_len, dtype=x.dtype), x[:shift_len]])
        return x

    @staticmethod
    def _apply_gain_and_noise(x: torch.Tensor, gain_range: tuple, snr_range: tuple) -> torch.Tensor:
        gain_db = gain_range[0] + (gain_range[1] - gain_range[0]) * torch.rand(1).item()
        snr_db = snr_range[0] + (snr_range[1] - snr_range[0]) * torch.rand(1).item()
        x = x * (10 ** (gain_db / 20))
        noise_power = x.pow(2).mean() / (10 ** (snr_db / 10))
        return x + torch.randn_like(x) * noise_power.sqrt()

    @staticmethod
    def _apply_random_dropout(x: torch.Tensor, drop_prob: bool, max_ratio: float = 0.02) -> torch.Tensor:
        if drop_prob:
            drop_len = int(max_ratio * len(x))
            if 0 < drop_len < len(x):
                start = torch.randint(0, len(x) - drop_len, (1,)).item()
                x = x.clone()
                x[start:start + drop_len] = 0.0
        return x

    def augment_audio_signal(self, x: torch.Tensor, augment_params: dict) -> torch.Tensor:
        x = self._apply_time_augment(x, augment_params)
        x = self._apply_gain_and_noise(x, gain_range=(-6, 6), snr_range=(15, 35))
        x = self._apply_random_dropout(x, augment_params["drop_prob"])
        return rms_normalize(x)

    def audio_mel(
        self,
        audio_path: str,
        n_mels: int = 80,
        augment_params: dict = None,
    ) -> torch.Tensor:
        """Load a source audio file and return a padded log-mel tensor."""
        audio = torch.from_numpy(whisper.load_audio(audio_path)).float()

        if augment_params is not None:
            audio = self.augment_audio_signal(audio, augment_params=augment_params)

        log_mel = whisper.log_mel_spectrogram(audio, n_mels)
        log_mel = whisper.pad_or_trim(log_mel, N_FRAMES)

        if self.augment:
            log_mel = self.apply_specaugment(log_mel)

        return log_mel
