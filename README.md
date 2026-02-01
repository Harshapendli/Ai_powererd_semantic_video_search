# VisualMind: AI-Powered Semantic Video Search

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-yellow)
![CLIP](https://img.shields.io/badge/AI-OpenAI%20CLIP-purple)

**VisualMind** is a local-first, privacy-focused search engine for raw video footage. It uses multimodal AI to understand **context, emotion, and cinematography**, allowing editors to find the perfect shot in seconds without manual tagging.

---

##  Key Features

### 1. Semantic Search (Beyond Keywords)
Search for concepts, not just filenames.
*   **Query:** *"Cinematic close-up of a tearful eye"*
*   **Result:** Finds the exact 3-second clip where this happens, even if the file is named `C0012.mp4`.

### 2.  Emotion Intelligence (Face-Aware)
Understand the emotional tone of your footage.
*   **Technology:** Uses **OpenCV** to detect faces and **CLIP** (Ensemble Prompting) to classify micro-expressions.
*   **Detected Emotions:** `Joy`, `Sadness`, `Fear`, `Anger`, `Tension`, `Neutral`.
*   **Smart Ranking:** Searching for *"sad scene"* prioritizes shots with actual sad facial expressions over generally dark scenes.

### 3.  Quality Guardrails
Stop wasting time on bad takes. The system automatically extracts technical metadata:
*   **Resolution:** Labels shots as `4K`, `1080p`, `720p`, or `480p`.
*   **Stability:** Filters out shaky or blurry footage.
*   **Ranking:** High-quality shots appear first in search results.

### 4.  Content-Aware Scene Splitting
*   **Granularity:** Returns the exact **cut** (e.g., `12.5s - 15.2s`) using **PySceneDetect**.
*   **No Fixed Segments:** Clips are cut exactly where the camera cuts, preserving narrative flow.

### 5.  Local & Private
*   **100% Offline:** No video data leaves your machine. Perfect for NDA-protected workflows.
*   **Comparison UI:** Side-by-side view of Original vs. Cleaned/Trimmed clips.

---

## 🛠️Tech Stack

*   **Core AI:** OpenAI CLIP (ViT-B/32)
*   **Computer Vision:** OpenCV (Haar Cascades, Motion Analysis)
*   **Scene Detection:** PySceneDetect
*   **Vector Database:** FAISS (Facebook AI Similarity Search)
*   **Backend:** FastAPI (High-performance Async API)
*   **Frontend:** Streamlit (Interactive Dashboard)

---

##  Installation

### Prerequisites
*   Python 3.10+
*   FFmpeg (installed and added to PATH)

### Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/visualmind.git
    cd visualmind
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

##  Usage

### 1. Add Videos
Place your `.mp4` or `.mkv` files in the `data/videos/` folder.

### 2. Start the Backend
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

### 3. Start the Frontend (New Terminal)
```bash
streamlit run frontend/app.py --server.port 8503
```

### 4. Build the Index
*   Open the app at `http://localhost:8503`.
*   Click **"Build / Rebuild Index"** in the top right.
*   *Note: First run takes time as it scans for faces and scenes.*

### 5. Search!
Try queries like:
*   *"High tension scene"*
*   *"Close up of a happy person at night"*
*   *"Wide landscape shot 4K"*

---

##  Project Structure

```
semantic_footage_search/
├── backend/
│   ├── main.py             # FastAPI entry point
│   ├── search.py           # Core Ranking Logic (Hybrid Semantic + Emotion)
│   ├── video_processing.py # OpenCV & PySceneDetect Logic
│   └── embeddings.py       # CLIP Model Handler
├── frontend/
│   └── app.py              # Streamlit UI
├── data/
│   ├── videos/             # RAW Input
│   ├── keyframes/          # Extracted JPEGs
│   └── metadata/           # shots.json (Rich Metadata)
└── requirements.txt
```

##  Contribution
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

##  License
[MIT](https://choosealicense.com/licenses/mit/)
