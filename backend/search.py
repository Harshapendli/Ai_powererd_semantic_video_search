from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .config import settings
from .embeddings import embed_images, embed_shots_mean_pool, embed_text
from .faiss_store import build_index_cosine, load_index, save_index, search_index
from .utils import ScenesMeta, read_json, write_json
from .video_processing import build_shots_and_frames

SHOT_TYPE_PROMPTS = ["wide shot", "medium shot", "close-up"]
EMOTION_PROMPTS = ["anger", "sadness", "joy", "fear", "neutral"]


def _load_metadata() -> ScenesMeta:
    return read_json(settings.METADATA_JSON_PATH, default=[])


def _save_metadata(shots: ScenesMeta) -> None:
    write_json(settings.METADATA_JSON_PATH, shots)


def _save_embeddings(emb: np.ndarray) -> None:
    settings.IMAGE_EMBEDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(settings.IMAGE_EMBEDS_PATH), emb)


def _enrich_shot_metadata(shots: List[Dict]) -> None:
    """Classify Emotion on FACE crops using CLIP."""
    pairs: List[tuple[int, str]] = []
    
    # 1. Collect Face Crops
    for i, s in enumerate(shots):
        meta = s.get("metadata", {})
        face_p = meta.get("face_path")
        if face_p:
            pairs.append((i, face_p))
            
    if not pairs:
        return

    try:
        paths = [p[1] for p in pairs]
        # Embed faces
        face_embs = embed_images(paths) # (N, 512)
        
        # Ensemble Prompts
        templates = [
            "face of a {} person",
            "expression of {}",
            "a {} face"
        ]
        
        # Build averaged emotion vectors
        # Shape: (E, 512)
        avg_emotion_vecs = []
        for em in settings.EMOTION_LABELS:
            # Embed all templates for this emotion
            texts = [t.format(em) for t in templates]
            vecs = np.stack([embed_text(t) for t in texts], axis=0)
            # Average and normalize
            mean_vec = np.mean(vecs, axis=0)
            mean_vec /= np.linalg.norm(mean_vec)
            avg_emotion_vecs.append(mean_vec)
            
        emotion_vecs = np.stack(avg_emotion_vecs, axis=0)
        
        # Classify
        # Scores = Face (N,512) @ Emotion.T (512, E) -> (N, E)
        sims = face_embs @ emotion_vecs.T
        
        for (shot_idx, _), row_sim in zip(pairs, sims):
            best_idx = int(np.argmax(row_sim))
            best_score = float(row_sim[best_idx])
            label = settings.EMOTION_LABELS[best_idx]
            
            # Additional Safety: 
            # If "neutral" (idx 0) is close to the winner, prefer neutral
            # Assume "neutral" is at index 0 (as per config change)
            neutral_score = row_sim[0]
            if label != "neutral" and (best_score - neutral_score < 0.015):
                label = "neutral"
                best_score = neutral_score 
            
            shots[shot_idx]["metadata"]["emotion"] = {
                "label": label,
                "confidence": best_score
            }
    except Exception as e:
        print(f"Enrichment error: {e}")


def build_or_rebuild_index() -> Dict[str, Any]:
    shots = build_shots_and_frames()
    _enrich_shot_metadata(shots) # ENABLED
    _save_metadata(shots)

    if len(shots) == 0:
        empty = build_index_cosine(np.zeros((0, 512), dtype=np.float32))
        save_index(empty, settings.FAISS_INDEX_PATH)
        _save_embeddings(np.zeros((0, 512), dtype=np.float32))
        return {"ok": True, "scenes_indexed": 0, "videos_found": 0}

    emb = embed_shots_mean_pool(shots)
    _save_embeddings(emb)

    index = build_index_cosine(emb)
    save_index(index, settings.FAISS_INDEX_PATH)

    videos_found = len({s["video_name"] for s in shots})
    return {"ok": True, "scenes_indexed": int(len(shots)), "videos_found": int(videos_found)}


def ensure_index_exists() -> Dict[str, Any]:
    index = load_index(settings.FAISS_INDEX_PATH)
    meta = _load_metadata()
    if index is None or not meta:
        return build_or_rebuild_index()
    return {"ok": True, "message": "Index already exists", "scenes_indexed": int(len(meta))}


def semantic_search(
    query: str, 
    top_k: int = 10,
    filters: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    query = (query or "").strip()
    filters = filters or {}
    
    if not query:
        return []

    index = load_index(settings.FAISS_INDEX_PATH)
    scenes = _load_metadata()
    if index is None or not scenes:
        return []

    q_vec = embed_text(query)  # (512,)
    
    # --- Emotion Extraction from Query ---
    query_emotion = None
    try:
        # Check query against emotion keywords
        em_vecs = np.stack([embed_text(e) for e in settings.EMOTION_LABELS], axis=0)
        q_sims = em_vecs @ q_vec
        best_e_idx = int(np.argmax(q_sims))
        if q_sims[best_e_idx] > 0.22: # Threshold for intent
            query_emotion = settings.EMOTION_LABELS[best_e_idx]
    except:
        pass

    # Fetch more candidates to allow for filtering
    search_k = min(len(scenes), top_k * 3)
    scores, ids = search_index(index, q_vec, search_k)

    results: List[Dict[str, Any]] = []
    
    min_quality = filters.get("min_quality", 0)
    min_relevance = filters.get("min_relevance", 0.18)
    editing_mode = filters.get("editing_mode", False)

    if editing_mode:
        if min_relevance < 0.28: min_relevance = 0.28

    candidates = []
    
    for score, idx in zip(scores.tolist(), ids.tolist()):
        if idx < 0 or idx >= len(scenes):
            continue
        
        # Relevance Threshold
        if score < min_relevance:
            continue
            
        s = scenes[idx]
        meta = s.get("metadata", {})
        q_score = meta.get("quality_score", 0)

        if q_score < min_quality:
            continue
            
        # --- Scoring Logic ---
        final_score = score
        
        # 1. Emotion Boost
        if query_emotion:
            shot_em = meta.get("emotion", {})
            if shot_em.get("label") == query_emotion:
                final_score += 0.15 # Boost matching emotion
            # else: maybe penalize?
            
        # 2. Editing Mode / Hybrid Ranker
        if editing_mode:
             # Hybrid: 70% Semantic, 30% Quality
             norm_quality = q_score / 100.0
             final_score = (final_score * 0.7) + (norm_quality * 0.3)
        
        candidates.append({
            "video_name": s["video_name"],
            "shot_id": int(s.get("shot_id", s.get("scene_id", 0))),
            "start_s": float(s["start_s"]),
            "end_s": float(s["end_s"]),
            "similarity": float(score),
            "final_score": float(final_score),
            "keyframe_rel": s.get("keyframe_rel"),
            "metadata": meta,
        })

    # Re-sort by final_score
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
    return candidates[:top_k]

