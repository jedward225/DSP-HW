import numpy as np
import torch
from typing import Optional
import torch.nn.functional as F

from src.retrieval.base import BaseRetriever
from src.dsp_core import log_melspectrogram


class SSIMRetriever(BaseRetriever):
    def __init__(
        self,
        name: str = 'SSIM',
        device: str = 'cpu',
        sr: int = 22050,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_frames: int = 216,
        k1: float = 0.01,
        k2: float = 0.03,
        window_size: int = 11,
        sigma: float = 1.5,
        padding: str = 'reflect',
        data_range: float = 1.0,
        chunk_size: int = 256,
    ):
        super().__init__(name=name, device=device, sr=sr)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_frames = n_frames
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.data_range = float(data_range)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.padding = str(padding)
        self.chunk_size = int(chunk_size)

        if self.window_size <= 0 or self.window_size % 2 == 0:
            raise ValueError('window_size must be a positive odd integer')

        if self.sigma <= 0:
            raise ValueError('sigma must be > 0')

        if self.chunk_size <= 0:
            raise ValueError('chunk_size must be > 0')

    @staticmethod
    def _gaussian_kernel_2d(
        window_size: int,
        sigma: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size // 2)
        g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        g = g / (g.sum() + 1e-12)
        kernel_2d = g[:, None] * g[None, :]
        kernel_2d = kernel_2d / (kernel_2d.sum() + 1e-12)
        return kernel_2d.view(1, 1, window_size, window_size)

    def _pad_for_conv(self, x: torch.Tensor, pad: int) -> torch.Tensor:
        if pad <= 0:
            return x
        mode = self.padding
        if mode == 'reflect':
            h, w = x.shape[-2], x.shape[-1]
            if h <= pad or w <= pad:
                mode = 'replicate'
        return F.pad(x, (pad, pad, pad, pad), mode=mode)

    def extract_features(self, waveform: torch.Tensor, sr: int = None) -> torch.Tensor:
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.from_numpy(waveform).float()

        waveform_np = waveform.detach().cpu().numpy().astype(np.float32)
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.reshape(-1)

        spec_db = log_melspectrogram(
            y=waveform_np,
            sr=sr if sr is not None else self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            ref='max',
        ).astype(np.float32)

        if spec_db.shape[1] < self.n_frames:
            pad = self.n_frames - spec_db.shape[1]
            spec_db = np.pad(spec_db, ((0, 0), (0, pad)), mode='edge')
        else:
            spec_db = spec_db[:, : self.n_frames]

        s_min = float(spec_db.min())
        s_max = float(spec_db.max())
        if s_max - s_min > 1e-10:
            spec = (spec_db - s_min) / (s_max - s_min)
        else:
            spec = np.zeros_like(spec_db)

        return torch.from_numpy(spec).to(self.device)

    def compute_distance(self, query_features: torch.Tensor, gallery_features: torch.Tensor) -> torch.Tensor:
        if query_features.dim() != 2:
            raise ValueError(f'Expected query_features to be 2D (H, W), got shape={tuple(query_features.shape)}')

        if gallery_features.dim() == 2:
            gallery_features = gallery_features.unsqueeze(0)
        if gallery_features.dim() != 3:
            raise ValueError(
                f'Expected gallery_features to be 3D (N, H, W), got shape={tuple(gallery_features.shape)}'
            )

        q = query_features.to(dtype=torch.float32)
        g_all = gallery_features.to(dtype=torch.float32)

        if g_all.shape[1:] != q.shape:
            raise ValueError(
                f'Gallery feature shape mismatch: query={tuple(q.shape)} gallery={tuple(g_all.shape[1:])}'
            )

        device = q.device
        dtype = q.dtype

        window = self._gaussian_kernel_2d(self.window_size, self.sigma, device=device, dtype=dtype)
        pad = self.window_size // 2

        q_img = q.unsqueeze(0).unsqueeze(0)
        q_pad = self._pad_for_conv(q_img, pad)

        mu_q = F.conv2d(q_pad, window)
        mu_q_sq = mu_q * mu_q
        sigma_q_sq = F.conv2d(q_pad * q_pad, window) - mu_q_sq
        sigma_q_sq = torch.clamp(sigma_q_sq, min=0.0)

        c1 = q_img.new_tensor((self.k1 * self.data_range) ** 2)
        c2 = q_img.new_tensor((self.k2 * self.data_range) ** 2)

        distances = []
        n = g_all.shape[0]
        for start in range(0, n, self.chunk_size):
            end = min(n, start + self.chunk_size)
            g = g_all[start:end].unsqueeze(1)
            g_pad = self._pad_for_conv(g, pad)

            mu_g = F.conv2d(g_pad, window)
            mu_g_sq = mu_g * mu_g

            sigma_g_sq = F.conv2d(g_pad * g_pad, window) - mu_g_sq
            sigma_g_sq = torch.clamp(sigma_g_sq, min=0.0)

            sigma_qg = F.conv2d(q_pad * g_pad, window) - (mu_q * mu_g)

            numerator = (2.0 * mu_q * mu_g + c1) * (2.0 * sigma_qg + c2)
            denominator = (mu_q_sq + mu_g_sq + c1) * (sigma_q_sq + sigma_g_sq + c2)

            ssim_map = numerator / (denominator + 1e-12)
            ssim = ssim_map.mean(dim=(1, 2, 3))
            dist = 1.0 - ssim
            distances.append(dist)

        return torch.cat(distances, dim=0)


def create_ssim_retriever(
    device: str = 'cpu',
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_frames: int = 216,
    **kwargs,
) -> SSIMRetriever:
    return SSIMRetriever(
        name='SSIM',
        device=device,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        n_frames=n_frames,
    )
