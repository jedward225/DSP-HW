#!/usr/bin/env python
# -*- coding: utf-8 -*-

import filecmp
import subprocess
import tempfile
from pathlib import Path
import shutil

import librosa
import numpy as np
import pandas as pd
import pytest


DATASET_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = DATASET_ROOT / 'audio'
META_PATH = DATASET_ROOT / 'meta' / 'esc50.csv'


@pytest.fixture(scope='module')
def recording_list():
    return sorted(p.name for p in AUDIO_DIR.iterdir() if p.is_file())


@pytest.fixture(scope='module')
def meta():
    return pd.read_csv(META_PATH)


def test_dataset_size(recording_list):
    assert len(recording_list) == 2000


def test_recordings(recording_list):
    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x):
            return x

    for recording in tqdm(recording_list):
        signal, rate = librosa.load(str(AUDIO_DIR / recording), sr=None, mono=False)

        assert rate == 44100
        assert len(signal.shape) == 1  # mono
        assert len(signal) == 220500  # 5 seconds
        assert np.max(signal) > 0
        assert np.min(signal) < 0
        assert np.abs(np.mean(signal)) < 0.2  # rough DC offset check


def test_previews(meta):
    convert = shutil.which('convert')
    if convert is None:
        pytest.skip("ImageMagick 'convert' not found")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        pytest.skip('matplotlib not installed')

    recordings = meta.groupby('target')['filename'].apply(
        lambda cat: cat.sample(1, random_state=20171207)
    ).reset_index()['filename']

    f, ax = plt.subplots(1, 1, sharey=False, sharex=False, figsize=(8, 2))

    with tempfile.TemporaryDirectory() as tmpdir:
        for index in range(len(recordings)):
            recording = recordings[index]
            signal = librosa.load(str(AUDIO_DIR / recording), sr=44100)[0]
            spec = librosa.feature.melspectrogram(signal, sr=44100, n_fft=2205, hop_length=441)
            spec = librosa.power_to_db(spec)

            category = meta[meta.filename == recording].category.values[0]

            ax.imshow(spec, origin='lower', interpolation=None, cmap='viridis', aspect=1.1)
            ax.set_title(f'{category} - {recording}', fontsize=11)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            f.tight_layout()
            plt.savefig(f'{tmpdir}/{index:02d}.png', bbox_inches='tight', dpi=72)

        png_paths = [str(p) for p in sorted(Path(tmpdir).glob('*.png'))]
        out_gif = DATASET_ROOT / '_esc50.gif'
        subprocess.check_call([convert, '-delay', '100', '-loop', '0', *png_paths, str(out_gif)])

    assert filecmp.cmp(DATASET_ROOT / 'esc50.gif', DATASET_ROOT / '_esc50.gif')
