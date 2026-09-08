"""
🚗 Vehicle Detection & Counting  —  Streamlit app powered by YOLOv8 (ultralytics)

Detects vehicles (car, bus, truck, motorcycle, bicycle) in an uploaded IMAGE or
VIDEO, draws bounding boxes, shows live per-class metric cards + totals, renders
a counting summary table, and (for video) offers the annotated video as a download.

Run:
    python -m streamlit run app.py --server.port 2005 --server.address 127.0.0.1

Notes:
    * yolov8n.pt (~6 MB) auto-downloads next to this file on first use.
    * CPU-only machine -> yolov8n (nano) keeps inference fast; video processing
      can still be slow, so frames are processed every `frame_step` frames and
      capped at `max_frames` (both adjustable in the sidebar) with a progress bar.
    * Annotated videos are encoded with the first available codec whose output
      OpenCV can read back (avc1/H.264 -> mp4v -> MJPG/avi), so the download is
      always a valid, playable file.
"""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# --------------------------------------------------------------------------- #
#  Paths & constants
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolov8n.pt"          # weights auto-downloaded on first use
SAMPLE_IMAGE = BASE_DIR / "bus.jpg"           # bundled sample image
OUTPUT_DIR = BASE_DIR / "generated"           # runtime annotated videos / uploads
DEFAULT_CONF = 0.30

# COCO class id -> pretty name  (vehicle classes only)
VEHICLE_CLASSES = {1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
VEHICLE_IDS = sorted(VEHICLE_CLASSES.keys())

VEHICLE_EMOJI = {
    "Bicycle": "🚲", "Car": "🚗", "Motorcycle": "🏍️",
    "Bus": "🚌", "Truck": "🚚",
}
CLASS_COLORS = {
    "Bicycle": "#4ade80", "Car": "#38bdf8", "Motorcycle": "#f472b6",
    "Bus": "#facc15", "Truck": "#fb923c",
}
TOTAL_COLOR = "#a78bfa"

IMG_EXTS = {"jpg", "jpeg", "png", "bmp", "webp"}
VID_EXTS = {"mp4", "avi", "mov", "mkv", "m4v", "webm"}

# (fourcc, container) tried in order when encoding the annotated video.
# Output must pass an OpenCV read-back check or the next candidate is tried.
CODEC_CANDIDATES = [("avc1", ".mp4"), ("mp4v", ".mp4"), ("MJPG", ".avi")]

# --------------------------------------------------------------------------- #
#  Detection helpers  (pure cv2/numpy — no Streamlit calls, unit-test friendly)
# --------------------------------------------------------------------------- #
def empty_counts() -> dict[str, int]:
    return {name: 0 for name in VEHICLE_CLASSES.values()}


def detect_image_bgr(model: YOLO, image_bgr: np.ndarray, conf: float = DEFAULT_CONF):
    """Run YOLOv8 on one BGR frame.

    Returns (annotated_bgr, per_class_counts, total_vehicles).
    """
    result = model.predict(
        image_bgr, conf=conf, classes=VEHICLE_IDS, imgsz=640, verbose=False
    )[0]
    counts = empty_counts()
    if result.boxes is not None and len(result.boxes) > 0:
        for c in result.boxes.cls.cpu().numpy().astype(int).tolist():
            if c in VEHICLE_CLASSES:
                counts[VEHICLE_CLASSES[c]] += 1
    return result.plot(), counts, int(sum(counts.values()))


def _clean_generated_dir(keep_prefixes: tuple[str, ...] = ()) -> None:
    """Remove previously generated files (never the one we are about to keep)."""
    if not OUTPUT_DIR.exists():
        return
    for p in OUTPUT_DIR.iterdir():
        if p.is_file() and not any(p.name.startswith(k) for k in keep_prefixes):
            try:
                p.unlink()
            except OSError:
                pass


def _encode_annotated_video(
    frame_jpgs: list[Path], out_fps: float, width: int, height: int
) -> tuple[Path, str]:
    """Encode cached annotated frames with the first codec whose output is readable.

    Returns (output_path, codec_used). Raises RuntimeError if none work.
    """
    ts = int(time.time())
    last_err = "no encoder available"
    for fourcc, ext in CODEC_CANDIDATES:
        cand = OUTPUT_DIR / f"annotated_{ts}{ext}"
        try:
            writer = cv2.VideoWriter(
                str(cand), cv2.VideoWriter_fourcc(*fourcc), out_fps, (width, height)
            )
            if not writer.isOpened():
                cand.unlink(missing_ok=True)
                last_err = f"{fourcc} encoder unavailable"
                continue
            for jp in frame_jpgs:
                frame = cv2.imread(str(jp))
                if frame is not None:
                    writer.write(frame)
            writer.release()

            # Read-back verification (Windows encoders can write unreadable files)
            cap = cv2.VideoCapture(str(cand))
            readable = bool(cap.isOpened())
            if readable:
                ok, _ = cap.read()
                readable = bool(ok)
            cap.release()
            if readable:
                return cand, fourcc
            last_err = f"{fourcc} output not readable"
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            last_err = f"{fourcc} failed: {exc}"
        cand.unlink(missing_ok=True)
    raise RuntimeError(f"Could not create a readable annotated video ({last_err}).")


def process_video_file(
    model: YOLO,
    video_path,
    conf: float = DEFAULT_CONF,
    frame_step: int = 2,
    max_frames: int = 300,
    progress_cb=None,
) -> dict:
    """Detect vehicles in a video file and write an annotated video into OUTPUT_DIR.

    Every `frame_step`-th frame is processed, up to `max_frames` frames total.
    `progress_cb(processed, planned)` is called after every processed frame.

    Returns a dict with the output path, totals / peak counts and stats.
    """
    video_path = Path(video_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clean stale outputs but KEEP the input file we are about to read
    # (it also lives in OUTPUT_DIR, e.g. saved uploads or test clips).
    _clean_generated_dir(keep_prefixes=(video_path.name,))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file — is the format supported?")

    try:
        fps_in = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        total_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        planned = max_frames
        if total_in > 0:
            planned = min(max_frames, int(math.ceil(total_in / max(1, frame_step))))
        planned = max(1, planned)

        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("The video file appears to be empty / unreadable.")
        height, width = frame.shape[:2]
        out_fps = max(1.0, fps_in / max(1, frame_step))

        frames_dir = OUTPUT_DIR / "_frames"
        frames_dir.mkdir(exist_ok=True)

        totals = empty_counts()
        peak = empty_counts()
        peak_total = 0
        processed = 0
        idx = 0
        jpgs: list[Path] = []

        # ---- pass 1: detection (slow, CPU) -> cache annotated frames as JPEG ----
        while processed < planned:
            if idx % max(1, frame_step) == 0:
                annotated, counts, frame_total = detect_image_bgr(model, frame, conf)
                jp = frames_dir / f"f{processed:05d}.jpg"
                cv2.imwrite(str(jp), annotated)
                jpgs.append(jp)
                for name, n in counts.items():
                    totals[name] += n
                    peak[name] = max(peak[name], n)
                peak_total = max(peak_total, frame_total)
                processed += 1
                if progress_cb is not None:
                    progress_cb(processed, planned)
            ok, frame = cap.read()
            idx += 1
            if not ok:
                break

        if processed == 0:
            raise RuntimeError("No frames could be read from the video.")

        # ---- pass 2: encode (fast, no model) with a readable codec --------------
        out_path, codec = _encode_annotated_video(jpgs, out_fps, width, height)

    finally:
        cap.release()
        shutil.rmtree(OUTPUT_DIR / "_frames", ignore_errors=True)

    return {
        "out_path": out_path,
        "codec": codec,
        "totals": totals,
        "peak": peak,
        "peak_total": peak_total,
        "frames_processed": processed,
        "planned_frames": planned,
        "total_frames": total_in,
        "fps_in": fps_in,
        "out_fps": out_fps,
        "frame_step": max(1, frame_step),
        "conf": conf,
    }


# --------------------------------------------------------------------------- #
#  Small UI building blocks
# --------------------------------------------------------------------------- #
def _css() -> str:
    return """
<style>
  .vc-hero { text-align:center; padding: .5rem 0 .1rem 0; }
  .vc-hero h1 {
    font-size: 2.6rem; margin: 0; letter-spacing: .5px;
    background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .vc-sub { text-align:center; color:#8b93a7; font-size:.95rem; margin-top:.35rem; }
  .vc-cards { display:flex; gap:10px; flex-wrap:wrap; justify-content:center; margin:.6rem 0 .4rem 0; }
  .vc-card {
    background:#11151f; border:1px solid #262c3f; border-radius:14px;
    padding:10px 16px; min-width:104px; text-align:center;
  }
  .vc-total { border-color:#6d5bd0; background: linear-gradient(180deg,#191334,#11151f); }
  .vc-label { color:#8b93a7; font-size:.82rem; margin-bottom:2px; }
  .vc-value { font-size:2rem; font-weight:800; line-height:1.15; }
  .vc-chip {
    display:inline-block; padding:1px 9px; border-radius:999px; margin-right:6px;
    font-size:.78rem; border:1px solid #2a2f45; color:#c3cadb;
  }
</style>
"""


def metric_cards_html(counts: dict[str, int], total: int) -> str:
    cards = []
    for name, n in counts.items():
        cards.append(
            f'<div class="vc-card"><div class="vc-label">{VEHICLE_EMOJI[name]} '
            f"{name}</div><div class=\"vc-value\" style=\"color:{CLASS_COLORS[name]}\">"
            f"{n}</div></div>"
        )
    cards.append(
        f'<div class="vc-card vc-total"><div class="vc-label">🚦 Total vehicles</div>'
        f'<div class="vc-value" style="color:{TOTAL_COLOR}">{total}</div></div>'
    )
    return f'<div class="vc-cards">{"".join(cards)}</div>'


def render_counts(counts: dict[str, int], total: int) -> None:
    st.markdown(metric_cards_html(counts, total), unsafe_allow_html=True)


def counts_dataframe(counts: dict[str, int], total: int) -> pd.DataFrame:
    rows = [
        {
            "Vehicle": f"{VEHICLE_EMOJI.get(n, '')} {n}",
            "Count": c,
            "Share": f"{100 * c / total:.1f}%" if total else "—",
        }
        for n, c in counts.items()
    ]
    rows.append({"Vehicle": "🚦 Total", "Count": total, "Share": "100.0%" if total else "—"})
    return pd.DataFrame(rows)


def render_summary_table(df: pd.DataFrame) -> None:
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Vehicle": st.column_config.TextColumn("Vehicle", width="medium"),
            "Count": st.column_config.NumberColumn("Count", format="%d"),
            "Share": st.column_config.TextColumn("Share"),
        },
    )


def render_hero() -> None:
    st.markdown(_css(), unsafe_allow_html=True)
    st.markdown(
        '<div class="vc-hero"><h1>🚦 Vehicle Detection & Counting</h1></div>'
        '<div class="vc-sub">YOLOv8 · car · bus · truck · motorcycle · bicycle '
        "· upload an image or a video</div>",
        unsafe_allow_html=True,
    )
    st.divider()


def annotate_info(counts: dict[str, int], total: int) -> str:
    present = [n for n, c in counts.items() if c > 0]
    if not present:
        return "No vehicles detected in the processed frames."
    return "Detected: " + "".join(
        f'<span class="vc-chip">{VEHICLE_EMOJI[n]} {n} × {counts[n]}</span>'
        for n in present
    )


# --------------------------------------------------------------------------- #
#  Main app
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="Vehicle Detection & Counting",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_hero()

    # ------------------------------------------------------------- sidebar -- #
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        model_choice = st.selectbox(
            "Model",
            options=["yolov8n"],
            format_func=lambda x: "yolov8n · nano (CPU-friendly)",
            help="yolov8n.pt (~6 MB) is auto-downloaded next to app.py on first use.",
        )
        conf = st.slider(
            "Confidence threshold", 0.10, 0.90, float(DEFAULT_CONF), 0.05,
            help="Detections below this confidence are discarded. 0.3 is a good default.",
        )
        st.markdown("##### 🎬 Video options (CPU speed)")
        frame_step = st.selectbox(
            "Process every Nth frame", options=[1, 2, 3], index=1,
            format_func=lambda x: {1: "Every frame", 2: "Every 2nd frame",
                                   3: "Every 3rd frame"}[x],
        )
        max_frames = st.slider("Max frames to process", 50, 900, 300, 50)
        st.caption(
            "→ up to ~" + str(int(min(max_frames, max_frames * frame_step)))
            + " source frames scanned. Larger videos are truncated for CPU speed."
        )
        st.divider()
        st.markdown("##### 🧠 About")
        st.caption(
            "Detects **car, bus, truck, motorcycle, bicycle** (COCO classes) with "
            "**YOLOv8n**. On images every detection is one vehicle. On videos counts "
            "are *detection-events* summed over the processed frames (a vehicle seen "
            "in many frames is counted each time); *Peak* = max concurrent in one frame."
        )

    # Model is cached for the whole session (loaded once, reused across reruns).
    model = st.cache_resource(_load_model)(model_choice)

    # ------------------------------------------------------------ uploader -- #
    upload = st.file_uploader(
        "📁 Upload an image or a video",
        type=sorted(IMG_EXTS | VID_EXTS),
        help="Images: JPG/PNG/BMP/WEBP · Videos: MP4/AVI/MOV/MKV",
    )

    if upload is not None:
        ext = Path(upload.name).suffix.lower().lstrip(".")
        if ext in IMG_EXTS:
            run_image(upload, model, conf)
        elif ext in VID_EXTS:
            run_video(upload, model, conf, frame_step, max_frames)
        else:
            st.warning(f"Unsupported file type: .{ext}")
        return

    # ------------------------------------------------------------ sample ----- #
    if SAMPLE_IMAGE.exists():
        c1, c2 = st.columns([1.1, 3])
        with c1:
            st.image(str(SAMPLE_IMAGE), caption="Bundled sample · bus.jpg", width=320)
        with c2:
            st.markdown(
                "**No file uploaded yet?** Try the bundled sample image below — "
                "a street scene with a bus from the Ultralytics repo."
            )
            if st.button("🎲 Detect on bundled bus.jpg", type="primary"):
                img = cv2.imread(str(SAMPLE_IMAGE))
                if img is None:
                    st.error("Could not read bus.jpg")
                else:
                    t0 = time.time()
                    annotated, counts, total = detect_image_bgr(model, img, conf)
                    dt = time.time() - t0
                    st.success(f"Done in {dt:.2f}s — {total} vehicle(s) detected.")
                    cA, cB = st.columns(2)
                    with cA:
                        st.image(img, channels="BGR", caption="Original")
                    with cB:
                        st.image(annotated, channels="BGR",
                                 caption=f"Annotated · {total} vehicle(s)")
                    render_counts(counts, total)
                    st.markdown(annotate_info(counts, total), unsafe_allow_html=True)
                    render_summary_table(counts_dataframe(counts, total))


def _load_model(model_key: str = "yolov8n") -> YOLO:
    """Load (and on first ever use, auto-download) the YOLOv8n weights."""
    return YOLO(str(MODEL_PATH))  # missing file -> ultralytics downloads yolov8n.pt


# --------------------------------------------------------------------------- #
#  Image workflow
# --------------------------------------------------------------------------- #
def run_image(upload, model: YOLO, conf: float) -> None:
    file_bytes = upload.read()
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not decode the uploaded image.")
        return

    with st.spinner("🔍 Running YOLOv8 detection…"):
        t0 = time.time()
        annotated, counts, total = detect_image_bgr(model, img, conf)
        dt = time.time() - t0

    st.success(f"Detected **{total}** vehicle(s) in {dt:.2f}s — "
               f"confidence ≥ {conf:.2f}")
    render_counts(counts, total)
    st.markdown(annotate_info(counts, total), unsafe_allow_html=True)

    cA, cB = st.columns(2)
    with cA:
        st.image(img, channels="BGR", caption="📷 Original image")
    with cB:
        st.image(annotated, channels="BGR",
                 caption=f"🖼️ Annotated · {total} vehicle(s)")

    ok, png = cv2.imencode(".png", annotated)
    if ok:
        st.download_button(
            "⬇️ Download annotated image (PNG)",
            data=png.tobytes(),
            file_name=f"annotated_{Path(upload.name).stem}.png",
            mime="image/png",
        )

    st.markdown("##### 📊 Counting summary")
    render_summary_table(counts_dataframe(counts, total))


# --------------------------------------------------------------------------- #
#  Video workflow
# --------------------------------------------------------------------------- #
def _save_upload(upload) -> Path:
    """Persist the uploaded bytes to a local file (cv2 needs a real path)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUTPUT_DIR / f"upload_{Path(upload.name).name}"
    with open(dst, "wb") as f:
        f.write(upload.getvalue())
    return dst


def run_video(upload, model: YOLO, conf: float, frame_step: int, max_frames: int) -> None:
    st.markdown(f"**🎬 Video:** `{upload.name}`  ·  confidence ≥ {conf:.2f} ·  "
                f"every {frame_step}‑th frame ·  cap {max_frames} frames")

    start_btn = st.button("▶️ Start vehicle detection", type="primary")
    placeholder = st.empty()

    if not start_btn:
        placeholder.info(
            "Click **▶️ Start vehicle detection** to process this video. "
            "YOLOv8n runs on CPU, so processing is capped to keep it quick "
            "(settings are in the sidebar)."
        )
        return

    # ---- kick off processing (long: progress bar, then stash the result) ---- #
    src_path = _save_upload(upload)
    progress = st.progress(0.0, text="Preparing video…")
    status = st.empty()

    def _cb(done: int, planned: int) -> None:
        progress.progress(min(1.0, done / planned),
                          text=f"Processing frame {done}/{planned} "
                               f"({100 * done // planned}%)…")
        status.caption(f"⏳ Keeping every {frame_step}‑th frame — this can take a "
                       f"minute on CPU.")

    try:
        result = process_video_file(
            model, src_path, conf=conf, frame_step=frame_step,
            max_frames=max_frames, progress_cb=_cb,
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        progress.empty()
        status.empty()
        st.error(f"Video processing failed: {exc}")
        return

    progress.empty()
    status.empty()

    total = int(sum(result["totals"].values()))
    truncated = (
        result["total_frames"] > 0
        and result["frames_processed"] * result["frame_step"] < result["total_frames"]
    )
    note = " (truncated)" if truncated else ""
    st.success(
        f"✅ Finished — {result['frames_processed']} frame(s) processed{note}. "
        f"**{total}** vehicle detection(s) across the video, peak "
        f"**{result['peak_total']}** vehicle(s) in one frame."
    )

    # ---- metric cards + summary table ------------------------------------ #
    st.markdown("##### 📊 Video counting summary")
    render_counts(result["totals"], total)

    rows = [
        {
            "Vehicle": f"{VEHICLE_EMOJI.get(n, '')} {n}",
            "Total detections": c,
            "Peak / frame": result["peak"][n],
        }
        for n, c in result["totals"].items()
    ]
    rows.append({
        "Vehicle": "🚦 All vehicles",
        "Total detections": total,
        "Peak / frame": result["peak_total"],
    })
    df = pd.DataFrame(rows)
    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Vehicle": st.column_config.TextColumn("Vehicle", width="medium"),
            "Total detections": st.column_config.NumberColumn("Total detections", format="%d"),
            "Peak / frame": st.column_config.NumberColumn("Peak / frame", format="%d"),
        },
    )
    st.caption(
        "*“Total detections” sums every processed frame (a vehicle seen across "
        "frames counts each time). “Peak / frame” is the most vehicles visible "
        "simultaneously in a single frame — a truer “how many are on screen” number.*"
    )

    # ---- annotated video player + download -------------------------------- #
    st.markdown("##### 🎥 Annotated video")
    out_path = result["out_path"]
    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    cA, cB = st.columns([3, 2])
    with cA:
        st.video(str(out_path))
    with cB:
        if total > 0:
            st.success(f"{total} vehicle detection(s) annotated.")
        else:
            st.warning("No vehicles found in the processed frames.")
        st.markdown(
            f"- Source: **{result['total_frames']} frames** @ "
            f"{result['fps_in']:.0f} fps ({result['total_frames'] / max(result['fps_in'], 1e-6):.1f}s)\n"
            f"- Processed: **{result['frames_processed']}** frames "
            f"(step {result['frame_step']}, conf {result['conf']:.2f})\n"
            f"- Output: **{result['out_fps']:.0f} fps** · codec "
            f"`{result['codec']}` · {size_mb:.1f} MB"
        )
        with open(out_path, "rb") as f:
            vid_bytes = f.read()
        st.download_button(
            "⬇️ Download annotated video",
            data=vid_bytes,
            file_name=f"annotated_{Path(upload.name).stem}{out_path.suffix}",
            mime="video/mp4" if out_path.suffix == ".mp4" else "video/x-msvideo",
        )
    st.caption("Annotated outputs are written to the `generated/` folder next to app.py.")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
