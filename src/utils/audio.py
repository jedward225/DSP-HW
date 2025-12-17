"""
Audio processing utilities.

This module provides audio processing utilities using only numpy/torch.
For operations requiring librosa/scipy, use dsp_core instead.
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional


class AudioProcessor:
    """
    Audio processing utilities using numpy/torch only.

    For librosa/scipy operations, use dsp_core module instead.
    """

    @staticmethod
    def normalize(
        waveform: Union[np.ndarray, torch.Tensor],
        method: str = 'peak'
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize audio waveform.

        Args:
            waveform: Input waveform
            method: Normalization method ('peak', 'rms', 'minmax')

        Returns:
            Normalized waveform
        """
        is_torch = isinstance(waveform, torch.Tensor)

        if method == 'peak':
            if is_torch:
                max_val = torch.abs(waveform).max()
                if max_val > 0:
                    waveform = waveform / max_val
            else:
                max_val = np.abs(waveform).max()
                if max_val > 0:
                    waveform = waveform / max_val

        elif method == 'rms':
            if is_torch:
                rms = torch.sqrt(torch.mean(waveform ** 2))
                if rms > 0:
                    waveform = waveform / rms
            else:
                rms = np.sqrt(np.mean(waveform ** 2))
                if rms > 0:
                    waveform = waveform / rms

        elif method == 'minmax':
            if is_torch:
                min_val = waveform.min()
                max_val = waveform.max()
                if max_val - min_val > 0:
                    waveform = 2 * (waveform - min_val) / (max_val - min_val) - 1
            else:
                min_val = waveform.min()
                max_val = waveform.max()
                if max_val - min_val > 0:
                    waveform = 2 * (waveform - min_val) / (max_val - min_val) - 1

        return waveform

    @staticmethod
    def pad_or_trim(
        waveform: Union[np.ndarray, torch.Tensor],
        target_length: int,
        pad_mode: str = 'constant',
        pad_value: float = 0.0
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Pad or trim waveform to target length.

        Args:
            waveform: Input waveform
            target_length: Target number of samples
            pad_mode: Padding mode ('constant', 'reflect', 'replicate')
            pad_value: Value for constant padding

        Returns:
            Waveform of exactly target_length samples
        """
        is_torch = isinstance(waveform, torch.Tensor)
        current_length = len(waveform)

        if current_length == target_length:
            return waveform

        elif current_length > target_length:
            # Trim
            return waveform[:target_length]

        else:
            # Pad
            pad_length = target_length - current_length

            if is_torch:
                if pad_mode == 'constant':
                    padding = torch.full((pad_length,), pad_value, dtype=waveform.dtype)
                    waveform = torch.cat([waveform, padding])
                elif pad_mode == 'reflect':
                    waveform = torch.nn.functional.pad(
                        waveform.unsqueeze(0).unsqueeze(0),
                        (0, pad_length),
                        mode='reflect'
                    ).squeeze()
                elif pad_mode == 'replicate':
                    waveform = torch.nn.functional.pad(
                        waveform.unsqueeze(0).unsqueeze(0),
                        (0, pad_length),
                        mode='replicate'
                    ).squeeze()
            else:
                if pad_mode == 'constant':
                    waveform = np.pad(
                        waveform, (0, pad_length),
                        mode='constant', constant_values=pad_value
                    )
                elif pad_mode == 'reflect':
                    waveform = np.pad(waveform, (0, pad_length), mode='reflect')
                elif pad_mode == 'replicate':
                    waveform = np.pad(waveform, (0, pad_length), mode='edge')

            return waveform

    @staticmethod
    def frame(
        waveform: Union[np.ndarray, torch.Tensor],
        frame_length: int,
        hop_length: int,
        center: bool = True,
        pad_mode: str = 'constant'
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Slice a waveform into overlapping frames.

        Args:
            waveform: Input waveform
            frame_length: Length of each frame in samples
            hop_length: Number of samples between frame starts
            center: If True, pad signal so frames are centered
            pad_mode: Padding mode

        Returns:
            Framed signal of shape (num_frames, frame_length)
        """
        is_torch = isinstance(waveform, torch.Tensor)

        if center:
            pad_amount = frame_length // 2
            if is_torch:
                waveform = torch.nn.functional.pad(
                    waveform.unsqueeze(0),
                    (pad_amount, pad_amount),
                    mode='constant' if pad_mode == 'constant' else 'reflect'
                ).squeeze(0)
            else:
                waveform = np.pad(waveform, pad_amount, mode=pad_mode)

        num_samples = len(waveform)
        num_frames = 1 + (num_samples - frame_length) // hop_length

        if is_torch:
            # Use unfold for efficient framing
            frames = waveform.unfold(0, frame_length, hop_length)
        else:
            # Create frame indices
            indices = np.arange(frame_length)[None, :] + \
                      np.arange(num_frames)[:, None] * hop_length
            frames = waveform[indices]

        return frames

    @staticmethod
    def add_noise(
        waveform: Union[np.ndarray, torch.Tensor],
        snr_db: float = 20.0
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Add Gaussian noise to waveform at specified SNR.

        Args:
            waveform: Input waveform
            snr_db: Signal-to-noise ratio in dB

        Returns:
            Noisy waveform
        """
        is_torch = isinstance(waveform, torch.Tensor)

        if is_torch:
            signal_power = torch.mean(waveform ** 2)
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        else:
            signal_power = np.mean(waveform ** 2)
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise = np.random.randn(*waveform.shape) * np.sqrt(noise_power)

        return waveform + noise

    @staticmethod
    def change_speed(
        waveform: Union[np.ndarray, torch.Tensor],
        rate: float = 1.0
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Change playback speed by resampling.

        Note: This is a simple implementation. For better quality,
        use librosa.effects.time_stretch via dsp_core.

        Args:
            waveform: Input waveform
            rate: Speed factor (>1 = faster, <1 = slower)

        Returns:
            Speed-changed waveform
        """
        is_torch = isinstance(waveform, torch.Tensor)
        original_length = len(waveform)
        new_length = int(original_length / rate)

        if is_torch:
            # Use linear interpolation
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            waveform = torch.nn.functional.interpolate(
                waveform, size=new_length, mode='linear', align_corners=False
            )
            waveform = waveform.squeeze()
        else:
            # Use numpy interpolation
            x_old = np.linspace(0, 1, original_length)
            x_new = np.linspace(0, 1, new_length)
            waveform = np.interp(x_new, x_old, waveform)

        return waveform

    @staticmethod
    def to_mono(waveform: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert stereo to mono by averaging channels.

        Args:
            waveform: Input waveform, shape (channels, samples) or (samples,)

        Returns:
            Mono waveform of shape (samples,)
        """
        if waveform.ndim == 1:
            return waveform

        is_torch = isinstance(waveform, torch.Tensor)

        if is_torch:
            return torch.mean(waveform, dim=0)
        else:
            return np.mean(waveform, axis=0)

    @staticmethod
    def to_torch(waveform: np.ndarray, device: str = 'cpu') -> torch.Tensor:
        """Convert numpy array to torch tensor."""
        return torch.from_numpy(waveform).float().to(device)

    @staticmethod
    def to_numpy(waveform: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array."""
        return waveform.detach().cpu().numpy()
