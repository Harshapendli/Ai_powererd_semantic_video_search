from __future__ import annotations

import time
from typing import Any, Dict, List

import requests
import streamlit as st


BACKEND = "http://127.0.0.1:8001"


def _post_index() -> Dict[str, Any]:
    r = requests.post(f"{BACKEND}/index", timeout=600)
    r.raise_for_status()
    return r.json()


def _search(q: str, k: int, min_quality: int = 0, min_relevance: float = 0.22, editing_mode: bool = False) -> Dict[str, Any]:
    params = {"q": q, "k": k, "min_quality": min_quality, "min_relevance": min_relevance, "editing_mode": editing_mode}
    r = requests.get(f"{BACKEND}/search", params=params, timeout=120)
    r.raise_for_status()
    return r.json()

def _cleanup_shot(video_name: str, start_s: float, end_s: float, shot_id: int) -> Dict[str, Any]:
    payload = {"video_name": video_name, "start_s": start_s, "end_s": end_s, "shot_id": shot_id}
    r = requests.post(f"{BACKEND}/cleanup_shot", json=payload, timeout=600)
    return r.json()

def _export_timeline(shot_paths: List[str]) -> Dict[str, Any]:
    payload = {"shot_paths": shot_paths}
    r = requests.post(f"{BACKEND}/export_timeline", json=payload, timeout=600)
    return r.json()


st.set_page_config(page_title="Semantic Footage Search", layout="wide")

# --- CUSTOM CSS FOR THEMES ---
theme_selection = st.sidebar.selectbox("🎨 UI Theme", ["3D Realistic (Dark)", "Minimalist (Light)", "Cyberpunk (Neon)"], index=0)

if theme_selection == "3D Realistic (Dark)":
    st.markdown("""
    <style>
        .stApp {
            background: radial-gradient(circle at 10% 20%, rgb(20, 20, 30) 0%, rgb(0, 0, 0) 90%);
            color: #e0e0e0;
        }
        div[data-testid="stVerticalBlock"] > div {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
         div.stButton > button {
            background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
            color: #ffffff;
            border: none;
            border-radius: 10px;
            box-shadow: 5px 5px 10px #0b0b0b, -5px -5px 10px #353535;
        }
    </style>
    """, unsafe_allow_html=True)

elif theme_selection == "Minimalist (Light)":
    st.markdown("""
    <style>
        .stApp {
            background: #f0f2f6;
            color: #31333F;
        }
        div[data-testid="stVerticalBlock"] > div {
            background: #ffffff;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
        }
    </style>
    """, unsafe_allow_html=True)

elif theme_selection == "Cyberpunk (Neon)":
    st.markdown("""
    <style>
        .stApp {
             background: #000000;
             color: #00ff41;
             font-family: 'Courier New', Courier, monospace;
        }
        div[data-testid="stVerticalBlock"] > div {
            border: 1px solid #00ff41;
            box-shadow: 0 0 10px #00ff41;
            background: #0d0d0d;
        }
        div.stButton > button {
            border: 1px solid #00ff41;
            background: black;
            color: #00ff41;
            text-transform: uppercase;
        }
        div.stButton > button:hover {
            background: #00ff41;
            color: black;
            box-shadow: 0 0 20px #00ff41;
        }
    </style>
    """, unsafe_allow_html=True)

st.title("AI-Powered Semantic Footage Search")

st.caption(
    "Shot-level search: content-aware boundaries (PySceneDetect), precise timestamps only where content appears. No fixed segments."
)

colA, colB, colC = st.columns([2, 1, 1])
with colA:
    q = st.text_input("Search query", value="emotional close-up at night")
with colB:
    k = st.number_input("Top-K", min_value=1, max_value=100, value=25, step=5, help="Show more results to find scenes from longer videos")
with colC:
    if st.button("Build / Rebuild Index", type="primary"):
        with st.spinner("Detecting shot boundaries (PySceneDetect) → Audio & Visual Embedding → Multimodal Fusion."):
            try:
                info = _post_index()
                st.success(f"Indexed {info.get('scenes_indexed', 0)} shots from {info.get('videos_found', 0)} video(s).")
                if "search_results" in st.session_state:
                    del st.session_state["search_results"]
            except Exception as e:
                st.error(f"Indexing failed: {e}")

# Advanced Filters Sidebar
with st.sidebar:
    st.header("Search Filters")
    min_quality = st.slider("Min Quality Score", 0, 100, 10, help="Filter out low-quality/blurry shots")
    min_relevance = st.slider("Strictness (Relevance)", 0.0, 0.5, 0.20, step=0.01, help="Higher = Fewer results but more accurate. >0.25 removes weak matches.")
    
    st.divider()
    editing_mode = st.toggle("🎥 Editing Mode", help="Prioritize High Quality shots. Apply stricter filtering.")
    
    st.caption("Filters apply strictly *after* semantic search.")
    
    st.divider()
    # Timeline Export
    if "selected_shots" not in st.session_state:
        st.session_state["selected_shots"] = []
    
    sel_count = len(st.session_state["selected_shots"])
    st.markdown(f"**Selection:** {sel_count} shots")
    if st.button("Export Timeline", disabled=(sel_count==0)):
        with st.spinner("Stitching timeline..."):
            res = _export_timeline(st.session_state["selected_shots"])
            if res.get("ok"):
                st.success("Timeline Created!")
                st.video(res["timeline_url"])
            else:
                st.error("Export failed.")
    
    if st.button("Clear Selection"):
        st.session_state["selected_shots"] = [] 
        st.rerun()

st.divider()

if st.button("Search", use_container_width=False):
    if not q.strip():
        st.warning("Enter a query.")
    else:
        # UI Fix: Clear previous clean states on new search
        keys_to_del = [del_k for del_k in st.session_state.keys() if del_k.startswith("clean_")]
        for del_k in keys_to_del:
            del st.session_state[del_k]

        with st.spinner("Searching..."):
            try:
                data = _search(q.strip(), int(k), min_quality, min_relevance, editing_mode)
                st.session_state["search_results"] = data
                st.session_state["search_query"] = q.strip()
            except Exception as e:
                st.error(f"Search failed: {e}")

# Show last search results
if "search_results" in st.session_state:
    data = st.session_state["search_results"]
    results: List[Dict[str, Any]] = data.get("results", [])
    st.subheader(f"Results for: \"{st.session_state.get('search_query', '')}\"")
    
    if editing_mode:
        st.caption("🎥 Editing Mode Active: Results sorted by Quality + Relevance.")
        
    if not results:
        st.info("No results. Try lowering strictness or searching for something else.")
    else:
        st.caption(f"{len(results)} result(s).")
        
        for r in results:
            shot_id = r.get("shot_id", 0)
            meta = r.get("metadata") or {}
            
            # Restore missing definitions
            start_s, end_s = r.get("start_s", 0), r.get("end_s", 0)
            video_name = r.get("video_name")
            thumb = r.get("thumbnail_url")
            
            # --- Classic Result Card Layout ---
            st.markdown("---")
            cols = st.columns([1, 3])
            
            with cols[0]:
                if thumb:
                    st.image(thumb, use_container_width=True)

            with cols[1]:
                st.subheader(f"{video_name} — Scene {shot_id}")
                st.write(f"**Range:** {start_s:.1f}s – {end_s:.1f}s")
                
                # Metadata Display
                q_score = meta.get("quality_score", 0)
                vid_q = meta.get("video_quality", "Unknown")
                
                emo = meta.get("emotion", {})
                emo_label = emo.get("label", "Neutral")
                emo_conf = emo.get("confidence", 0.0)
                
                score_color = "green" if q_score > 80 else "orange" if q_score > 50 else "red"
                
                # Badges
                st.markdown(f"""
                **Quality:** :{score_color}[{q_score}/100] | **Res:** `{vid_q}` | **Relevance:** `{r.get('similarity'):.4f}`
                """)
                if emo_label != "Neutral":
                    st.markdown(f"**Emotion:** `{emo_label.title()}` ({emo_conf:.2f})")
                
                if editing_mode:
                    f_score = r.get("final_score", 0)
                    st.caption(f"Hybrid Score: {f_score:.4f}")

                feedback = meta.get("editor_feedback", [])
                if feedback:
                    st.write(" ".join([f"`{tag}`" for tag in feedback]))

            # --- VIDEO & COMPARISON UI ---
            from urllib.parse import quote
            safe_vid_name = quote(video_name)
            v_url = f"{BACKEND}/videos/{safe_vid_name}"
            
            clean_key = f"clean_{shot_id}"
            has_clean = f"clean_url_{shot_id}" in st.session_state

            # Smart Trim Button
            if st.button("✨ Smart Trim / Clean", key=clean_key):
                 with st.spinner("Analyzing & Trimming..."):
                    c_res = _cleanup_shot(video_name, start_s, end_s, shot_id)
                    if c_res.get("ok"):
                       st.success("Trimmed!")
                       st.session_state[f"clean_url_{shot_id}"] = c_res["cleaned_url"]
                       st.session_state[f"clean_path_{shot_id}"] = c_res["path"]
                       st.rerun()

            # Comparison Container
            with st.container():
                 if has_clean:
                     # Side-by-Side Comparison
                     c1, c2 = st.columns(2)
                     
                     with c1:
                         st.markdown("##### 🔴 Original")
                         st.video(v_url, start_time=int(start_s))
                     
                     with c2:
                         st.markdown("##### 🟢 Cleaned")
                         st.video(st.session_state[f"clean_url_{shot_id}"])
                     
                     # Timeline Selection
                     c_path = st.session_state[f"clean_path_{shot_id}"]
                     is_sel = (c_path in st.session_state["selected_shots"])
                     
                     if st.checkbox("✅ Add to Timeline", value=is_sel, key=f"sel_{shot_id}"):
                         if not is_sel: st.session_state["selected_shots"].append(c_path)
                     else:
                         if is_sel: st.session_state["selected_shots"].remove(c_path)

                 else:
                     # Single View (Original)
                     with st.expander("▶️ Play Original Clip (Full Size)", expanded=False):
                         st.video(v_url, start_time=int(start_s))
            

st.divider()
st.caption("Demo tip: try queries like “crowd celebration”, “rainy chase”, “dark alley at night”, “wide landscape shot”.")

