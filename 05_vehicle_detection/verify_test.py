"""
Quick offline sanity test for the Vehicle Detection app pipeline.

Imports app.py as a plain module (its Streamlit UI only runs under
`streamlit run`, so nothing here needs a browser) and verifies:
  1. yolov8n detects >= 1 vehicle on the bundled bus.jpg sample.
  2. The video pipeline writes an annotated MP4 for a short synthetic clip.

Run:  python verify_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import app as vd  # noqa: E402


def test_image() -> int:
    model = YOLO(str(vd.MODEL_PATH))
    img = cv2.imread(str(vd.SAMPLE_IMAGE))
    assert img is not None, f"sample image not found: {vd.SAMPLE_IMAGE}"

    t0 = time.time()
    annotated, counts, total = vd.detect_image_bgr(model, img, conf=0.3)
    dt = time.time() - t0

    print(f"[IMAGE]  bus.jpg shape={img.shape}  infer={dt:.2f}s")
    print(f"[IMAGE]  per-class counts: {counts}")
    print(f"[IMAGE]  total vehicles : {total}")
    assert total > 0, "FAIL: expected >= 1 vehicle on bus.jpg"
    assert annotated.shape == img.shape, "FAIL: annotated image shape mismatch"
    print("[IMAGE]  PASS (annotated image returned, vehicles > 0)")


def test_video() -> None:
    model = YOLO(str(vd.MODEL_PATH))
    # 16-frame synthetic clip, 10 fps, 320x240 with a big dark "car-ish" blob.
    # MJPG + .avi: reliably readable by OpenCV on Windows. NOTE: the processing
    # pipeline cleans OUTPUT_DIR first, so the input name must be whitelisted
    # (process_video_file keeps video_path.name) — verified by this test.
    clip = BASE / "generated" / "_smoke_test.avi"
    clip.parent.mkdir(parents=True, exist_ok=True)
    h, w, fps, n = 240, 320, 10, 16
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for i in range(n):
        frame = np.full((h, w, 3), 120, np.uint8)
        x = 30 + i * 8
        cv2.rectangle(frame, (x, 120), (x + 90, 200), (40, 40, 200), -1)  # blue blob
        writer.write(frame)
    writer.release()
    assert clip.exists(), "FAIL: could not write synthetic clip"

    t0 = time.time()
    result = vd.process_video_file(
        model, clip, conf=0.3, frame_step=2, max_frames=6,
        progress_cb=lambda done, planned: None,
    )
    dt = time.time() - t0

    print(f"[VIDEO]  processed={result['frames_processed']}  "
          f"total_frames={result['total_frames']}  "
          f"totals={result['totals']}  time={dt:.1f}s")
    out = result["out_path"]
    assert out.exists() and out.stat().st_size > 0, "FAIL: no annotated MP4 produced"
    assert result["frames_processed"] > 0, "FAIL: no frames were processed"
    print(f"[VIDEO]  PASS (annotated MP4 written: {out.name}, "
          f"{out.stat().st_size / 1e3:.0f} KB)")
    # cleanup smoke artifacts
    clip.unlink(missing_ok=True)
    out.unlink(missing_ok=True)


if __name__ == "__main__":
    test_image()
    test_video()
    print("\nALL CHECKS PASSED ✔")
