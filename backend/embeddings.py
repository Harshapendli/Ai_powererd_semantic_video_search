from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image


@lru_cache(maxsize=1)
def load_clip_model():
    # OpenAI CLIP package: `import clip`
    import clip  # type: ignore

    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    return model, preprocess, device, clip


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


@torch.inference_mode()
def embed_images(image_paths: List[str], batch_size: int = 16) -> np.ndarray:
    model, preprocess, device, _clip = load_clip_model()

    feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
        image_input = torch.stack(imgs).to(device)
        image_features = model.encode_image(image_input).float()
        feats.append(image_features.cpu().numpy())

    if not feats:
        return np.zeros((0, 512), dtype=np.float32)

    emb = np.vstack(feats).astype(np.float32)
    emb = _l2_normalize(emb)
    return emb


def embed_shots_mean_pool(shots: List[dict], batch_size: int = 16) -> np.ndarray:
    """
    For each shot: encode ALL representative frames (start/mid/end), mean-pool, L2 normalize.
    Returns one embedding per shot, shape (n_shots, 512). Precision-first: shot-level semantics.
    """
    if not shots:
        return np.zeros((0, 512), dtype=np.float32)
    all_frame_paths: List[str] = []
    shot_boundaries: List[int] = [0]
    for s in shots:
        # FAST MODE: Only use the keyframe (middle frame)
        # fps = s.get("frame_paths") or [s.get("keyframe_path", "")]
        fps = [s.get("keyframe_path", "")]
        fps = [p for p in fps if p]
        all_frame_paths.extend(fps)
        shot_boundaries.append(shot_boundaries[-1] + len(fps))
    if not all_frame_paths:
        return np.zeros((len(shots), 512), dtype=np.float32)
    frame_emb = embed_images(all_frame_paths, batch_size=batch_size)
    shot_embs = []
    for i in range(len(shots)):
        start_idx = shot_boundaries[i]
        end_idx = shot_boundaries[i + 1]
        
        # 1. Visual Embedding (Mean of frames)
        if end_idx <= start_idx:
            vis_emb = np.zeros(512, dtype=np.float32)
        else:
            chunk = frame_emb[start_idx:end_idx]
            vis_emb = chunk.mean(axis=0).astype(np.float32)
            # Normalize visual part first?
            vis_emb = vis_emb / np.maximum(np.linalg.norm(vis_emb), 1e-12)

        # Multimodal Fusion REMOVED
        shot_embs.append(vis_emb)

    out = np.stack(shot_embs, axis=0)
    return _l2_normalize(out)


@torch.inference_mode()
def embed_text(text: str) -> np.ndarray:
    model, _preprocess, device, clip_mod = load_clip_model()
    tokens = clip_mod.tokenize([text]).to(device)
    text_features = model.encode_text(tokens).float().cpu().numpy().astype(np.float32)
    text_features = text_features / np.maximum(np.linalg.norm(text_features, axis=1, keepdims=True), 1e-12)
    return text_features[0]

