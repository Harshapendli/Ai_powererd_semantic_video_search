from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import ensure_data_dirs, settings
from .search import build_or_rebuild_index, ensure_index_exists, semantic_search


app = FastAPI(title="Semantic Footage Search (Local Demo)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    ensure_data_dirs()
    # Don't force rebuild on startup; just ensure something exists.
    ensure_index_exists()


# Serve extracted keyframes for the Streamlit UI
ensure_data_dirs()
app.mount("/static", StaticFiles(directory=str(settings.KEYFRAMES_DIR)), name="static")
settings.CLEANED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static_cleaned", StaticFiles(directory=str(settings.CLEANED_DIR)), name="static_cleaned")
app.mount("/videos", StaticFiles(directory=str(settings.VIDEOS_DIR)), name="videos")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.post("/index")
def rebuild_index() -> Dict[str, Any]:
    return build_or_rebuild_index()


@app.get("/search")
@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    k: int = Query(25, ge=1, le=100),
    min_quality: int = Query(0, ge=0, le=100),
    min_relevance: float = Query(0.22, ge=0.0, le=1.0),
    editing_mode: bool = Query(False)
    # has_dialogue: bool = Query(False) # REMOVED
) -> Dict[str, Any]:
    filters = {"min_quality": min_quality, "min_relevance": min_relevance, "editing_mode": editing_mode}
    results = semantic_search(q, top_k=k, filters=filters)
    # Attach thumbnail URL for UI
    for r in results:
        rel = (r.get("keyframe_rel") or "").lstrip("/")
        r["thumbnail_url"] = f"http://127.0.0.1:8001/static/{rel}" if rel else None
    return {"query": q, "k": k, "results": results}


# --- V3 Editor Features ---

from pydantic import BaseModel

class CleanupRequest(BaseModel):
    video_name: str
    start_s: float
    end_s: float
    shot_id: int

@app.post("/cleanup_shot")
def cleanup_shot(req: CleanupRequest) -> Dict[str, Any]:
    from .editing import detect_content_bounds_visual, perform_smart_trim
    
    video_path = settings.VIDEOS_DIR / req.video_name
    if not video_path.exists():
        return {"ok": False, "error": "Video file not found"}
        
    out_name = f"clean_shot_{req.shot_id}_{int(req.start_s)}_{int(req.end_s)}.mp4"
    out_path = settings.CLEANED_DIR / out_name
    
    ok = perform_smart_trim(video_path, out_path, req.start_s, req.end_s)
    
    if ok:
        rel_path = f"cleaned/{out_name}"
        url = f"http://127.0.0.1:8001/static_cleaned/{out_name}"
        return {"ok": True, "cleaned_url": url, "path": str(out_path)}
    return {"ok": False, "error": "FFmpeg failed"}


class TimelineRequest(BaseModel):
    shot_paths: List[str]

@app.post("/export_timeline")
def export_timeline(req: TimelineRequest) -> Dict[str, Any]:
    from .editing import create_rough_cut_timeline
    
    paths = [Path(p) for p in req.shot_paths]
    out_name = "rough_cut_timeline.mp4"
    out_path = settings.CLEANED_DIR / out_name
    
    ok = create_rough_cut_timeline(paths, out_path)
    if ok:
         url = f"http://127.0.0.1:8001/static_cleaned/{out_name}"
         return {"ok": True, "timeline_url": url}
    return {"ok": False, "error": "Export failed"}

