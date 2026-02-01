"""
Audio Intelligence Module.
Handles extraction, speech-to-text (Whisper), and signal quality analysis.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import librosa
    import numpy as np
    import whisper
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: Audio dependencies (librosa/whisper) not installed. Audio features disabled.")

from .config import settings

# Global model cache
_WHISPER_MODEL = None

def get_whisper_model():
    global _WHISPER_MODEL
    if not AUDIO_AVAILABLE:
        return None
    if _WHISPER_MODEL is None:
        # 'tiny', 'base', 'small', 'medium', 'large'
        # Using 'base' as a balance for local CPU
        print(f"Loading Whisper model ({settings.WHISPER_MODEL})...")
        try:
            _WHISPER_MODEL = whisper.load_model(settings.WHISPER_MODEL)
        except Exception as e:
            print(f"Failed to load Whisper: {e}")
    return _WHISPER_MODEL


def extract_audio_features(
    video_path: Path, start_s: float, end_s: float
) -> Dict[str, Any]:
    """
    Extract audio from the specific video segment, compute metrics, and transcribe.
    Returns: {
        "transcript": str,
        "audio_score": float (0-100),
        "is_silent": bool,
        "has_dialogue": bool
    }
    """
    if not AUDIO_AVAILABLE:
         return {"transcript": "", "audio_score": 0.0, "is_silent": True, "has_dialogue": False}

    path_str = str(video_path)
    try:
        # Load audio segment using librosa
        # offset=start_s, duration=(end_s - start_s)
        # librosa uses ffmpeg backend
        y, sr = librosa.load(path_str, offset=start_s, duration=(end_s - start_s), sr=16000)
    except Exception as e:
        print(f"Audio extraction failed for {path_str}: {e}")
        return {"transcript": "", "audio_score": 0.0, "is_silent": True, "has_dialogue": False}

    if len(y) == 0:
        return {"transcript": "", "audio_score": 0.0, "is_silent": True, "has_dialogue": False}

    # 1. Basic Metrics
    rms = float(np.sqrt(np.mean(y**2)))
    # Simple silence check
    is_silent = rms < 0.005 # Threshold needs tuning

    # 2. Transcribe (if not silent)
    transcript = ""
    if not is_silent:
        model = get_whisper_model()
        # Whisper can take the numpy array directly (at 16k Hz)
        # Note: pad/trim handled by whisper
        result = model.transcribe(y, fp16=False) # fp16=False for CPU safety
        transcript = result.get("text", "").strip()

    # 3. Audio Quality Score (Stage 6)
    # Simple heuristic: penalize ultra-quiet or clipping
    score = 100.0
    if is_silent:
        score = 50.0 # Neutral if silent
    else:
        # Check clipping
        if np.max(np.abs(y)) > 0.98:
            score -= 20 # Clipping
        
        # Check strictly noise (zero crossing rate high + low spectral flatness?)
        # For now, just transcription confidence? 
        # Whisper doesn't easily give confidence per whole segment in simple API, 
        # but we can assume if transcript is long, it's dialogue.
        pass
    
    return {
        "transcript": transcript,
        "audio_score": score,
        "is_silent": is_silent,
        "has_dialogue": len(transcript) > 2,
        "volume_rms": rms
    }
