"""
Quick test script for CLAP zero-shot classification on ESC-50.
Expected accuracy: ~92% on Fold 5

Reference: [Ref-2] 苏慧学等 - Zero-shot: 92.25% (Fold5)
"""

import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import laion_clap
import librosa
from tqdm import tqdm
import pandas as pd

# ESC-50 class names (in order of target 0-49)
ESC50_CLASSES = [
    "dog", "rooster", "pig", "cow", "frog",
    "cat", "hen", "insects", "sheep", "crow",
    "rain", "sea waves", "crackling fire", "crickets", "chirping birds",
    "water drops", "wind", "pouring water", "toilet flush", "thunderstorm",
    "crying baby", "sneezing", "clapping", "breathing", "coughing",
    "footsteps", "laughing", "brushing teeth", "snoring", "drinking sipping",
    "door knock", "mouse click", "keyboard typing", "door wood creaks",
    "can opening", "washing machine", "vacuum cleaner", "clock alarm",
    "clock tick", "glass breaking",
    "helicopter", "chainsaw", "siren", "car horn", "engine",
    "train", "church bells", "airplane", "fireworks", "hand saw"
]

# Better prompts for zero-shot
PROMPTS = [
    "This is a sound of {}",
    "A sound of {}",
    "The sound of {}",
]


def main():
    print("=" * 70)
    print("CLAP Zero-Shot Classification on ESC-50")
    print("=" * 70)

    # Load CLAP model
    print("\n[1] Loading CLAP model...")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny')
    model.load_ckpt()  # Load default checkpoint
    print("    Model loaded!")

    # Compute text embeddings for all classes
    print("\n[2] Computing text embeddings...")
    all_text_embeddings = []
    for prompt_template in PROMPTS:
        prompts = [prompt_template.format(cls) for cls in ESC50_CLASSES]
        text_embed = model.get_text_embedding(prompts, use_tensor=True)
        text_embed = torch.nn.functional.normalize(text_embed, dim=-1)
        all_text_embeddings.append(text_embed)

    # Average across prompts
    text_embeddings = torch.stack(all_text_embeddings).mean(dim=0)
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
    print(f"    Text embeddings shape: {text_embeddings.shape}")

    # Load ESC-50 metadata
    print("\n[3] Loading ESC-50 Fold 5 (test set)...")
    data_root = PROJECT_ROOT / "ESC-50"
    meta_path = data_root / "meta" / "esc50.csv"
    metadata = pd.read_csv(meta_path)

    # Filter Fold 5
    test_meta = metadata[metadata['fold'] == 5].reset_index(drop=True)
    print(f"    Test samples: {len(test_meta)}")

    # Evaluate
    print("\n[4] Running zero-shot evaluation...")
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for idx in tqdm(range(len(test_meta)), desc="Evaluating"):
        row = test_meta.iloc[idx]
        audio_path = data_root / "audio" / row['filename']
        label = row['target']

        # Load audio (CLAP expects 48kHz)
        audio, sr = librosa.load(str(audio_path), sr=48000)

        # Get audio embedding
        # CLAP expects audio in shape (batch, samples)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)

        with torch.no_grad():
            audio_embed = model.get_audio_embedding_from_data(
                x=audio_tensor,
                use_tensor=True
            )
            audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)

        # Compute similarity
        similarity = torch.matmul(audio_embed, text_embeddings.T)
        pred = similarity.argmax(dim=-1).item()

        all_preds.append(pred)
        all_labels.append(label)

        if pred == label:
            correct += 1
        total += 1

    accuracy = 100.0 * correct / total

    print("\n" + "=" * 70)
    print(f"Results:")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Expected (Ref-2): ~92.25%")
    print("=" * 70)

    # Per-class accuracy
    print("\n[5] Per-class accuracy (top 5 worst):")
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    class_acc = {}
    for i, cls_name in enumerate(ESC50_CLASSES):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc[cls_name] = (all_preds[mask] == all_labels[mask]).mean() * 100

    # Sort by accuracy
    sorted_acc = sorted(class_acc.items(), key=lambda x: x[1])
    for cls_name, acc in sorted_acc[:5]:
        print(f"    {cls_name:20s}: {acc:.1f}%")

    print("\n[6] Top 5 best classes:")
    for cls_name, acc in sorted_acc[-5:]:
        print(f"    {cls_name:20s}: {acc:.1f}%")

    return accuracy


if __name__ == "__main__":
    main()
