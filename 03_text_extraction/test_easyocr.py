"""Test easyocr on the sample image (first run downloads detection + recognition models)."""
import sys, time

img = "C:/Users/Admin/course_projects/03_text_extraction/sample_text.png"
t0 = time.time()
import easyocr
print(f"[{time.time()-t0:.1f}s] easyocr imported, creating Reader(['en'], gpu=False) ...", flush=True)
reader = easyocr.Reader(["en"], gpu=False, verbose=True)
print(f"[{time.time()-t0:.1f}s] Reader ready, running readtext ...", flush=True)
results = reader.readtext(img)
print(f"[{time.time()-t0:.1f}s] DONE. {len(results)} boxes found", flush=True)
for box, text, conf in results:
    print(f"  conf={conf:.2f} text={text!r}")
