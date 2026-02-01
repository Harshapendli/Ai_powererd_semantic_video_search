"""
Editor Assistance Module.
Handles non-destructive "Smart Trimming" and "Timeline Export".
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio_ffmpeg

from .config import settings


def get_ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def detect_content_bounds_visual(video_path: Path, threshold: float = 15.0) -> Tuple[float, float, bool]:
    """
    Scans start and end of video for black/static frames.
    Returns (start_sec, end_sec, was_trimmed).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0, 0.0, False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0.0

    if duration < 1.0:
        cap.release()
        return 0.0, duration, False

    # 1. Scan Start (up to 20% or 5s)
    clean_start = 0.0
    limit_frames = int(min(frame_count * 0.2, fps * 5))
    
    for i in range(limit_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.mean() > threshold:
            clean_start = i / fps
            break
            
    # 2. Scan End (from back, up to 20% or 5s)
    clean_end = duration
    # Seeking from end is slow, so we jump carefully
    end_limit_frames = int(min(frame_count * 0.2, fps * 5))
    start_search_frame = max(0, frame_count - end_limit_frames)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_search_frame)
    # We read forward and remember the last "good" frame timestamp
    last_good_time = clean_end # Default to end
    
    for i in range(start_search_frame, frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_time = i / fps
        if gray.mean() > threshold:
            last_good_time = curr_time
            
    clean_end = last_good_time
    # Add a small buffer to end to avoid cutting last movement
    clean_end = min(duration, clean_end + 0.1)

    cap.release()
    
    # Safety: If we trimmed too much (>80%), assume detection failed and return original
    if (clean_end - clean_start) < (duration * 0.2):
        return 0.0, duration, False
        
    was_trimmed = (clean_start > 0.1) or (clean_end < duration - 0.1)
    return clean_start, clean_end, was_trimmed


def perform_smart_trim(
    video_path: Path, 
    output_path: Path, 
    start_s: float, 
    end_s: float
) -> bool:
    """
    Uses FFmpeg to losslessly trim (re-encode for precision if needed) the video.
    We prefer re-encoding for frame accuracy in "Smart Trim" mode.
    """
    if output_path.exists():
        return True # Cached

    settings.CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    
    duration = end_s - start_s
    ffmpeg = get_ffmpeg_exe()
    
    # Command: ffmpeg -ss start -i input -t duration -c:v libx264 -preset ultrafast -c:a copy out.mp4
    # We re-encode video (-c:v libx264) to ensure the cut is visually precise (GOP issues with stream copy).
    # Audio copy is safe.
    cmd = [
        ffmpeg, "-y",
        "-ss", str(start_s),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", # Re-encode audio to be safe 
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def create_rough_cut_timeline(shot_paths: List[Path], output_path: Path) -> bool:
    """
    Concatenates list of video files into a single timeline.
    Uses generic Concat Demuxer.
    """
    if not shot_paths:
        return False
        
    ffmpeg = get_ffmpeg_exe()
    list_file = output_path.parent / "concat_list.txt"
    
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for p in shot_paths:
                # Escape path for ffmpeg
                safe_path = str(p.resolve()).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")
        
        # Concat command
        cmd = [
            ffmpeg, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False
    finally:
        if list_file.exists():
            list_file.unlink()
