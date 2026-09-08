"""Test the exact ocr_core dispatch path used by app.py on the real sample image."""
import sys, time
sys.path.insert(0, "C:/Users/Admin/course_projects/03_text_extraction")
import numpy as np
from PIL import Image
import ocr_core

# Load sample like the app does
img = np.array(Image.open("C:/Users/Admin/course_projects/03_text_extraction/sample_text.png").convert("RGB"))

t0 = time.time()
res = ocr_core.ocr_image(img)
print(f"dispatch took {res['elapsed']:.1f}s wall {time.time()-t0:.1f}s")
print("backend:", res["backend"])
print("error:", res["error"])
print("note:", res["note"])
print("n_boxes:", res["n_boxes"], "avg_conf:", res["avg_conf"])
print("lines:")
for ln in res["lines"]:
    print("  |", ln)
assert res["backend"] == "easyocr", "expected easyocr primary backend"
assert len(res["lines"]) > 0, "expected non-empty text"
assert "EasyOCR Test 123" in res["text"] and "2026" in res["text"], res["text"]
print("PASS: easyocr primary path verified on real image")

# Also verify preprocess + draw_regions don't crash
proc = ocr_core.preprocess(img, grayscale=True, threshold=True, upscale=True, scale=2)
print("preprocess ->", proc.shape, proc.dtype)
ann = ocr_core.draw_regions(proc, res["boxes"])
print("draw_regions ->", ann.shape)
print("PASS: preprocessing + region drawing OK")
