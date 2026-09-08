"""Headless verification of the face swap engine on the bundled sample pair."""
import os
import sys

import cv2
import numpy as np

import face_swap_engine as fse

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "samples", "source_man.jpg")
TGT = os.path.join(ROOT, "samples", "target_woman.jpg")
OUT = os.path.join(ROOT, "result_test_man2woman.png")

def mean_abs_diff(a, b):
    a = cv2.resize(a, (b.shape[1], b.shape[0])) if a.shape != b.shape else a
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())

def main():
    with open(SRC, "rb") as f:
        src_bytes = f.read()
    with open(TGT, "rb") as f:
        tgt_bytes = f.read()

    src_img = fse.decode_image(src_bytes)
    tgt_img = fse.decode_image(tgt_bytes)
    print(f"mediapipe status: {fse.mediapipe_status()}")
    print(f"source {src_img.shape[1]}x{src_img.shape[0]}  target {tgt_img.shape[1]}x{tgt_img.shape[0]}")

    # face-detection smoke check
    from face_swap_engine import _mesh_points, _haar_faces
    ms, mt = _mesh_points(src_img), _mesh_points(tgt_img)
    print("mesh src:", ms is not None, " mesh tgt:", mt is not None,
          " haar src:", _haar_faces(src_img) is not None,
          " haar tgt:", _haar_faces(tgt_img) is not None)

    result, info = fse.swap_faces(src_bytes, tgt_bytes)
    print("swap info:", info)
    cv2.imwrite(OUT, result)

    assert os.path.isfile(OUT) and os.path.getsize(OUT) > 1000, "output not written"
    d_src = mean_abs_diff(result, src_img)
    d_tgt = mean_abs_diff(result, tgt_img)
    print(f"output: {OUT} ({os.path.getsize(OUT)} bytes)")
    print(f"mean |result-source| = {d_src:.2f}   mean |result-target| = {d_tgt:.2f}")
    assert d_src > 3 and d_tgt > 3, "result too similar to an input - swap failed?"
    print("OK: swapped output produced and differs from both inputs.")

if __name__ == "__main__":
    sys.exit(main())
