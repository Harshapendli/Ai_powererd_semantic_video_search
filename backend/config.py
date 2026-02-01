from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Shot detection: content-aware boundaries only (no fixed durations)
    # Min shot length (seconds) to avoid tiny fragments
    MIN_SHOT_LENGTH_S: float = 0.5
    
    # Emotion Project
    EMOTION_LABELS = ["neutral", "joy", "sadness", "anger", "fear", "tension"]
    
    # Resolution Standards (Height)
    RES_480P: int = 480
    RES_720P: int = 720
    RES_1080P: int = 1080
    RES_4K: int = 2160

    # Data dirs (relative to project root)
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    VIDEOS_DIR: Path = DATA_DIR / "videos"
    KEYFRAMES_DIR: Path = DATA_DIR / "keyframes"
    CLEANED_DIR: Path = DATA_DIR / "cleaned"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    METADATA_DIR: Path = DATA_DIR / "metadata"

    # Persisted artifacts (shot-level)
    FAISS_INDEX_PATH: Path = EMBEDDINGS_DIR / "faiss.index"
    IMAGE_EMBEDS_PATH: Path = EMBEDDINGS_DIR / "shot_embeddings.npy"
    METADATA_JSON_PATH: Path = METADATA_DIR / "shots.json"

    # CLIP
    CLIP_MODEL_NAME: str = "ViT-B/32"
    # Audio
    WHISPER_MODEL: str = "base"


settings = Settings()


def ensure_data_dirs() -> None:
    settings.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    settings.KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)
    settings.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    settings.METADATA_DIR.mkdir(parents=True, exist_ok=True)
