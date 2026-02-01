"""
Run the full semantic search demo: health -> index -> search.
Execute from semantic_footage_search/ with:  python run_demo.py
Backend must be running on http://127.0.0.1:8000
"""
import sys
import requests

BASE = "http://127.0.0.1:8001"

def main():
    print("1. Health check...")
    try:
        r = requests.get(f"{BASE}/health", timeout=10)
        print("   ", r.json())
    except Exception as e:
        print("   FAIL:", e)
        print("   Start backend: python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000")
        sys.exit(1)

    print("2. Build index (may take 2-5 min first time for CLIP)...")
    try:
        print("   (This may take a while for large videos...)")
        r = requests.post(f"{BASE}/index", timeout=3600)
        data = r.json()
        print("   ", data)
        if data.get("scenes_indexed", 0) == 0:
            print("   Put MP4 files in data/videos/ then run again.")
    except Exception as e:
        print("   FAIL:", e)
        sys.exit(1)

    print("3. Search: 'emotional close-up at night'")
    try:
        r = requests.get(f"{BASE}/search", params={"q": "emotional close-up at night", "k": 5}, timeout=60)
        d = r.json()
        print("   Query:", d["query"], "| Results:", len(d["results"]))
        for i, x in enumerate(d["results"][:5]):
            print("   ", i + 1, x["video_name"], "%.1fs-%.1fs" % (x["start_s"], x["end_s"]), "score=%.4f" % x["similarity"])
    except Exception as e:
        print("   FAIL:", e)
        sys.exit(1)

    print("\nDone. Open Streamlit: streamlit run frontend/app.py")

if __name__ == "__main__":
    main()
