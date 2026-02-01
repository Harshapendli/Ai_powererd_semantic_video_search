from __future__ import annotations

from pathlib import Path
from typing import Tuple

import faiss  # type: ignore
import numpy as np


def build_index_cosine(embeddings: np.ndarray) -> faiss.Index:
    """
    Cosine similarity via inner product on L2-normalized vectors.
    """
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    d = int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] > 0 else 512
    index = faiss.IndexFlatIP(d)
    if embeddings.shape[0] > 0:
        index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index | None:
    if not path.exists():
        return None
    return faiss.read_index(str(path))


def search_index(index: faiss.Index, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    if query_vec.dtype != np.float32:
        query_vec = query_vec.astype(np.float32)
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    scores, ids = index.search(query_vec, top_k)
    return scores[0], ids[0]

