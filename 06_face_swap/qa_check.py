"""Numeric QA of the face swap result + high-res run (no vision needed)."""
import os
import cv2
import numpy as np
import face_swap_engine as fse
from face_swap_engine import _mesh_points, _mask_from_points

ROOT = os.path.dirname(os.path.abspath(__file__))

def hull_region_stats(result, tgt, pts_tgt):
    hull = _mask_from_points(tgt.shape, pts_tgt)
    inner = hull > 0
    outer = hull == 0
    d_in = np.abs(result.astype(np.int16) - tgt.astype(np.int16))[inner].mean()
    d_out = np.abs(result.astype(np.int16) - tgt.astype(np.int16))[outer].mean()
    return d_in, d_out

src = fse.decode_image(open(os.path.join(ROOT, "samples", "source_man.jpg"), "rb").read())
tgt = fse.decode_image(open(os.path.join(ROOT, "samples", "target_woman.jpg"), "rb").read())
res = fse.decode_image(open(os.path.join(ROOT, "result_test_man2woman.png"), "rb").read())

pts_t = _mesh_points(tgt)
d_in, d_out = hull_region_stats(res, tgt, pts_t)
print(f"[128px run] mean diff INSIDE target face hull = {d_in:.1f}, OUTSIDE = {d_out:.2f}")
print("  -> change concentrated in face region:", d_in > 3 * d_out and d_out < 3)

# skin-tone sanity: pasted face should look like a face (not blank/black hole)
hull = _mask_from_points(tgt.shape, pts_t)
inner_pix = res[hull > 0]
mean_lum = inner_pix.mean()
print(f"  mean luminance of pasted region = {mean_lum:.1f} (0-255; >60 means not a black hole)")

# 512px upscaled run: exercises warp at 4x resolution
src_big = cv2.resize(src, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
tgt_big = cv2.resize(tgt, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
res_big, info = fse.swap_faces(src_big, tgt_big)
print(f"[512px run] method={info['method']} time={info['time_ms']}ms out={info['out_shape']}")
cv2.imwrite(os.path.join(ROOT, "result_test_512px.png"), res_big)
pts_tb = _mesh_points(tgt_big)
din_b, dout_b = hull_region_stats(res_big, tgt_big, pts_tb)
print(f"[512px run] mean diff INSIDE hull = {din_b:.1f}, OUTSIDE = {dout_b:.2f}")
print("OK 512px: change concentrated in face:", din_b > 3 * dout_b)

# side-by-side QA sheet
sheet = np.hstack([cv2.resize(src, (256, 256)), cv2.resize(res, (256, 256)), cv2.resize(tgt, (256, 256))])
cv2.putText(sheet, "SOURCE", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
cv2.putText(sheet, "RESULT", (261, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
cv2.putText(sheet, "TARGET", (517, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
cv2.imwrite(os.path.join(ROOT, "qa_sheet.png"), sheet)
print("QA sheet written: qa_sheet.png (source | result | target)")
