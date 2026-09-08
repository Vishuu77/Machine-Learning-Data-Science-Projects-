# 🎭 Face Swap Studio (course project 06 — port 2006)

Streamlit app that pastes a **SOURCE** person's face onto a **TARGET** person's photo.

## Launch
```bash
cd C:/Users/Admin/course_projects/06_face_swap
python -m streamlit run app.py --server.port 2006 --server.address 127.0.0.1
```

## How it works
- **Primary:** mediapipe `FaceLandmarker` (tasks API, v1.0.1, model in
  `mediapipe_models/face_landmarker.task`) detects 478 landmarks on both faces →
  shared Delaunay triangulation → per-triangle affine warp maps the source face
  texture onto the target's face geometry → per-channel color transfer matches
  skin tone/lighting → `cv2.seamlessClone` blends the face hull into the photo.
- **Fallback (no mediapipe):** OpenCV Haar cascades + eye detection → similarity
  warp → elliptical mask → seamless clone.
- CPU-only; working resolution capped at 1100 px so swaps take ~0.3–1.5 s.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (dark theme, sample photos, download button) |
| `face_swap_engine.py` | Engine — no streamlit imports, headless-testable |
| `test_engine.py` | Headless verification (sample pair → `result_test_man2woman.png`) |
| `qa_check.py` | Numeric QA (change confined to face hull, high-res run) |
| `samples/` | `source_man.jpg`, `target_woman.jpg` (real photos, 128 px) |
| `mediapipe_models/face_landmarker.task` | Landmark model (3.7 MB) |

## Limitations
Frontal, well-lit faces with one clear face per photo. Output keeps the target's
pose/hair/background and the source's facial texture; strong head tilt, heavy
occlusion, or big lighting differences reduce quality.
