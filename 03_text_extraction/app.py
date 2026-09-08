"""📝 OCR Text Extractor - Streamlit app.

Upload an image -> easyocr (CPU) extracts every text region (Windows built-in OCR
used as an automatic fallback). Optional preprocessing: grayscale, threshold,
upscale. Side-by-side image / extracted-text view, detected regions overlay,
download results as .txt.

Run:  python -m streamlit run app.py --server.port 2003 --server.address 127.0.0.1
"""
import hashlib
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import ocr_core

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(PROJECT_DIR, "sample_text.png")

# --------------------------------------------------------------------------- #
# Page config + styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="OCR Text Extractor",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
  .block-container { padding-top: 1.4rem; }
  #MainMenu, footer { visibility: hidden; }
  .hero { text-align: center; padding: 0.1rem 0 0.7rem 0; }
  .hero h1 {
    font-size: 2.6rem; margin-bottom: 0.1rem;
    background: linear-gradient(90deg, #00e5a0, #4fc3f7, #ff8a65);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero p { color: #9aa4b2; font-size: 1.02rem; margin-top: 0.2rem; }
  .cards { display: flex; gap: 0.9rem; justify-content: stretch; margin: 0.2rem 0 0.9rem 0; }
  .card {
    flex: 1; background: #1a1d29; border: 1px solid #2c3142; border-radius: 12px;
    padding: 0.75rem 1rem; text-align: center;
  }
  .card .label { color: #8a93a6; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
  .card .value { color: #ffffff; font-size: 1.55rem; font-weight: 700; line-height: 1.25; }
  .card .sub { color: #5ce6b0; font-size: 0.75rem; }
  .engine-pill {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px;
    background: #123127; color: #5ce6b0; border: 1px solid #1f5c46;
    font-size: 0.8rem; margin: 0.25rem 0 0.6rem 0;
  }
  .pill-amber { background:#33290f; color:#ffc24b; border-color:#6b5215; }
  .section-label { color: #8a93a6; font-size: 0.85rem; letter-spacing: 0.08em;
                   text-transform: uppercase; margin: 0.4rem 0 0.35rem 0; }
  div[data-testid="stFileUploaderDropzone"] { border-color: #2c3142; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    """
<div class="hero">
  <h1>📝 OCR Text Extractor</h1>
  <p>Upload a picture · every text region is read out · copy or download as <b>.txt</b></p>
</div>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def metric_cards(chars, words, regions, conf):
    conf_txt = "—" if conf is None else f"{conf * 100:.0f}%"
    html = (
        '<div class="cards">'
        f'<div class="card"><div class="label">Characters</div><div class="value">{chars}</div></div>'
        f'<div class="card"><div class="label">Words</div><div class="value">{words}</div></div>'
        f'<div class="card"><div class="label">Text regions</div><div class="value">{regions}</div></div>'
        f'<div class="card"><div class="label">Avg. confidence</div><div class="value">{conf_txt}</div>'
        "<div class=\"sub\">easyocr score</div></div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def load_sample_bytes():
    with open(SAMPLE_PATH, "rb") as fh:
        return fh.read()


def rgb_from_bytes(data):
    return np.array(Image.open(__import__("io").BytesIO(data)).convert("RGB"))


def fp_of(img_bytes, gray, thresh, upscale, scale):
    h = hashlib.sha1(img_bytes)
    h.update(("|%s|%s|%s|%d" % (gray, thresh, upscale, scale)).encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Sidebar: input + preprocessing controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🖼️ 1 · Source image")
    uploaded = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"],
        label_visibility="collapsed",
    )
    sample_clicked = st.button("✨ Try the bundled sample image", width="stretch")

    st.markdown("### ⚙️ 2 · Preprocessing")
    gray = st.toggle("Grayscale", value=False, help="Convert to grayscale before OCR")
    thresh = st.toggle("Threshold (B/W)", value=False,
                       help="Adaptive binarisation - auto-enables grayscale. Helps scans / photos")
    upscale = st.toggle("Upscale 2×", value=False,
                        help="Resize image up before OCR - helps small / low-res text")
    st.caption("Tip: clean black-on-white images usually OCR best with no preprocessing.")

    st.markdown("### 🧠 3 · Engine")
    st.caption("1️⃣ easyocr (CPU) — models already cached in `~/.EasyOCR`")
    st.caption("2️⃣ Windows built-in OCR — automatic fallback")
    st.markdown("---")
    st.caption("Built with ❤️ · Streamlit + easyocr + OpenCV")

# Resolve image bytes: uploaded file wins, else sample image (kept across reruns).
if uploaded is not None:
    img_bytes = uploaded.getvalue()
    st.session_state["img_bytes"] = img_bytes
    st.session_state["img_name"] = uploaded.name
elif sample_clicked:
    st.session_state["img_bytes"] = load_sample_bytes()
    st.session_state["img_name"] = "sample_text.png"

if "img_bytes" not in st.session_state:
    # ---- Landing state: no image yet ------------------------------------- #
    st.markdown(
        """
    <div style="text-align:center; padding: 5rem 0 2rem 0;">
      <div style="font-size:4.2rem;">🖼️</div>
      <div style="color:#9aa4b2; font-size:1.15rem; margin-top:0.6rem;">
        No image loaded yet.<br>
        Use the uploader or the sample button in the sidebar 👈
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.stop()

img_bytes = st.session_state["img_bytes"]
img_name = st.session_state["img_name"]

# --------------------------------------------------------------------------- #
# OCR (fingerprint-cached: only re-runs when image or preprocessing changes)
# --------------------------------------------------------------------------- #
fp = fp_of(img_bytes, gray, thresh, upscale, 2)
if st.session_state.get("fp") != fp:
    img_rgb = rgb_from_bytes(img_bytes)
    processed = ocr_core.preprocess(img_rgb, gray, thresh, upscale, 2)
    with st.spinner("🔍 Reading text…"):
        res = ocr_core.ocr_image(processed)
    st.session_state.update(
        fp=fp, img_rgb=img_rgb, processed=processed, res=res,
        # free the easyocr lock only via cache_resource; nothing else to clean
    )
    if res["error"]:
        st.session_state["last_error"] = res["error"]
else:
    img_rgb = st.session_state["img_rgb"]
    processed = st.session_state["processed"]
    res = st.session_state["res"]

text = res["text"]
conf = res["avg_conf"]
backend = res["backend"]
n_boxes = res["n_boxes"] if res["boxes"] is not None else 0
chars = len(text)
words = len(text.split())

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
if res["error"]:
    st.error("⚠️ OCR failed on every backend: " + res["error"])
    st.stop()

metric_cards(chars, words, n_boxes, conf)

pill_class = "engine-pill" if backend == "easyocr" else "engine-pill pill-amber"
engine_label = (
    "Engine: easyocr 1.7.2 (CPU)"
    if backend == "easyocr"
    else "Engine: Windows built-in OCR (fallback)"
)
st.markdown(
    f'<span class="{pill_class}">{engine_label}</span> '
    f'<span style="color:#6b7686; font-size:0.85rem;">· {res["elapsed"]:.1f} s · '
    f'{img_name}</span>',
    unsafe_allow_html=True,
)
if res["note"] and backend == "windows":
    st.caption(res["note"])

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">🖼️ Image · detected regions</div>',
                unsafe_allow_html=True)
    show_boxes = st.checkbox("Outline detected text regions", value=True)
    if res["boxes"] is not None and show_boxes:
        display = ocr_core.draw_regions(processed, res["boxes"])
    else:
        display = processed
    st.image(display, width="stretch")
    if gray or thresh or upscale:
        pre = " + ".join(
            [n for n, on in (("grayscale", gray), ("threshold", thresh), ("upscale 2×", upscale)) if on]
        )
        st.caption(f"Preprocessed: {pre} · right-click image to save")
    if backend == "windows":
        st.caption("ℹ️ Windows OCR reports text only - region outlines come from easyocr, "
                   "so no boxes are drawn on the fallback path.")

with right:
    st.markdown('<div class="section-label">📄 Extracted text</div>',
                unsafe_allow_html=True)
    if text:
        st.code(text, language=None)
        base = os.path.splitext(img_name)[0]
        st.download_button(
            "⬇️ Download as .txt",
            data=text.encode("utf-8"),
            file_name=f"{base}_ocr.txt",
            mime="text/plain",
            width="stretch",
        )
        with st.expander("🔎 Per-region detail (easyocr)"):
            if res["boxes"] is not None:
                for i, (line) in enumerate(res["lines"], 1):
                    st.markdown(f"`{i:02d}` {line}")
                st.caption(f"{n_boxes} text regions, average confidence "
                           f"{conf * 100:.0f}%" if conf is not None else "n/a")
            else:
                st.caption("Not available on the Windows OCR fallback path.")
    else:
        st.info("No text was detected in this image.")
        st.caption("Try enabling the upscale / threshold toggles, or use a clearer image.")
