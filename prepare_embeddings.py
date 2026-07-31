"""Build a vocabulary-aligned 100d GloVe matrix with coverage metadata."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import urllib.request

import numpy as np
import torch

import config
from dataset import Vocab, clean_text, load_splits

GLOVE_URL = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
    "glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz"
)
DIMENSION = 100


def build_embedding_artifact(
    output_path: Path = config.ARTIFACTS_DIR / "glove_100d_l2dtnh.pt",
) -> dict:
    train_df, _, _ = load_splits()
    vocab = Vocab.build(
        (clean_text(text) for text in train_df["text"]),
        max_size=config.MAX_VOCAB_SIZE,
        min_freq=config.MIN_FREQ,
    )
    cache_path = config.ARTIFACTS_DIR / "downloads" / "glove-wiki-gigaword-100.gz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        print(f"Downloading pretrained embeddings from {GLOVE_URL}")
        urllib.request.urlretrieve(GLOVE_URL, cache_path)

    generator = np.random.default_rng(config.SEED)
    matrix = generator.normal(0.0, 0.05, (len(vocab), DIMENSION)).astype(
        np.float32
    )
    matrix[0] = 0.0
    matched = set()
    wanted = set(vocab.stoi) - {"<pad>", "<unk>"}
    with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip().split()
            if len(fields) != DIMENSION + 1 or fields[0] not in wanted:
                continue
            matrix[vocab.stoi[fields[0]]] = np.asarray(
                fields[1:],
                dtype=np.float32,
            )
            matched.add(fields[0])

    coverage = {
        "source": "GloVe Wikipedia 2014 + Gigaword 5",
        "url": GLOVE_URL,
        "dimension": DIMENSION,
        "vocabulary_tokens": len(wanted),
        "matched_tokens": len(matched),
        "token_coverage": len(matched) / max(len(wanted), 1),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "matrix": torch.from_numpy(matrix),
            "vocab_stoi": vocab.stoi,
            "coverage": coverage,
        },
        output_path,
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(coverage, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(coverage, indent=2))
    return coverage


if __name__ == "__main__":
    build_embedding_artifact()
