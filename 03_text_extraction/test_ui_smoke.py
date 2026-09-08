"""Headless UI smoke test: run app.py through Streamlit's AppTest harness."""
import sys
sys.path.insert(0, "C:/Users/Admin/course_projects/03_text_extraction")
from streamlit.testing.v1 import AppTest

# 1) Landing state - no image selected yet
at = AppTest.from_file("C:/Users/Admin/course_projects/03_text_extraction/app.py", default_timeout=240)
at.run()
assert not at.exception, f"landing exception: {at.exception}"
print("landing state OK - no exceptions")

# 2) Click "Try the bundled sample image" -> triggers real OCR through the UI
at.button(key=None)  # no-op safety
sample_btns = [b for b in at.button if "sample" in (b.label or "").lower()]
assert sample_btns, f"sample button not found; buttons={[b.label for b in at.button]}"
sample_btns[0].click().run()
assert not at.exception, f"sample-run exception: {at.exception}"
print("sample-run state OK - no exceptions")

# Pull extracted text out of the st.code block and metric values from HTML cards
codes = [c.value for c in at.code] if hasattr(at, "code") else []
text = codes[0] if codes else ""
print("--- extracted text from UI code block ---")
print(text)
assert "EasyOCR Test 123" in text and "2026" in text, f"unexpected text: {text!r}"
print("PASS: AppTest end-to-end UI run returned expected OCR text")
