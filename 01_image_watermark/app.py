"""Watermark Studio — a Streamlit app for non-destructive image watermarking.

Upload an image, then either burn a configurable TEXT watermark
(font size, opacity, rotation, colour, position or full tiling) or blend a
small LOGO image (scale, position, opacity) into it. The original pixels are
never modified — the result is composited on a copy and offered as a PNG
download, with an original-vs-watermarked side-by-side preview.

Run with:
    python -m streamlit run app.py --server.port 2001
"""

from __future__ import annotations

import os
import re

import streamlit as st
from PIL import Image

import watermark_engine as wm

# --------------------------------------------------------------------------- #
# Page setup & dark theme
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Watermark Studio",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    .stApp {
        background:
            radial-gradient(1200px 620px at 12% -10%, rgba(56, 105, 170, 0.18), transparent 60%),
            radial-gradient(1000px 520px at 108% 8%, rgba(130, 74, 190, 0.14), transparent 55%),
            #0a0e15;
        color: #dbe5f1;
    }
    .hero {
        display: flex; align-items: center; gap: 18px;
        background: linear-gradient(135deg, #101a2b, #0c1220);
        border: 1px solid #22334d; border-radius: 16px;
        padding: 18px 24px; margin: 4px 0 18px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .hero-emoji { font-size: 42px; }
    .hero-title { font-size: 26px; font-weight: 800; color: #f1f6fd; letter-spacing: .3px; }
    .hero-sub { color: #8ea4c2; font-size: 14px; margin-top: 3px; }
    .brand {
        display: flex; align-items: center; gap: 10px;
        padding: 4px 2px 10px; border-bottom: 1px solid #1d2a3e; margin-bottom: 12px;
    }
    .brand-emoji { font-size: 26px; }
    .brand-name { font-size: 17px; font-weight: 800; color: #eef4fc; }
    .brand-tag { font-size: 12px; color: #7187a5; }
    .section-label {
        font-size: 12px; font-weight: 700; text-transform: uppercase;
        letter-spacing: .8px; color: #7e94b3; margin: 14px 0 6px;
    }
    .dscard-row { display: flex; gap: 14px; flex-wrap: wrap; margin: 4px 0 18px; }
    .dscard {
        flex: 1; min-width: 150px;
        background: linear-gradient(160deg, #0f1726, #0c1322);
        border: 1px solid #22344e; border-radius: 12px; padding: 12px 16px;
    }
    .dsicon { font-size: 20px; }
    .dslabel { color: #7890ae; font-size: 11px; text-transform: uppercase;
               letter-spacing: .7px; margin-top: 6px; }
    .dsvalue { color: #edf4fd; font-size: 21px; font-weight: 700; margin-top: 2px; }
    .shot-title { font-size: 14px; font-weight: 700; color: #b9c9dd; margin-bottom: 6px; }
    .empty-card {
        text-align: center; padding: 46px 24px; margin: 6px 0 18px;
        background: linear-gradient(160deg, #0e1626, #0b111d);
        border: 1px dashed #2b3f5e; border-radius: 16px;
    }
    .empty-emoji { font-size: 52px; }
    .empty-title { font-size: 22px; font-weight: 800; color: #eef4fd; margin-top: 8px; }
    .empty-sub { color: #8ba0bd; font-size: 14px; margin-top: 4px; max-width: 640px;
                 margin-left: auto; margin-right: auto; }
    .step-grid { display: flex; gap: 14px; flex-wrap: wrap; justify-content: center;
                 margin: 22px auto 0; max-width: 860px; }
    .step-card {
        flex: 1; min-width: 200px; text-align: left;
        background: #0d1524; border: 1px solid #1f2f47; border-radius: 12px;
        padding: 14px 16px;
    }
    .step-num { color: #5f8fd9; font-weight: 800; font-size: 13px; }
    .step-txt { color: #b9c8da; font-size: 13px; margin-top: 5px; line-height: 1.5; }
    .warn-card {
        background: rgba(214, 158, 46, 0.12); border: 1px solid rgba(214, 158, 46, 0.45);
        border-radius: 12px; padding: 14px 18px; color: #ffd9a0; margin: 8px 0 16px;
        font-size: 15px;
    }
    .footer {
        margin-top: 30px; padding-top: 14px; border-top: 1px solid #1b2637;
        color: #5d718e; font-size: 13px; text-align: center;
    }
    [data-testid="stFileUploader"] section {
        background: #0d1424; border: 1px dashed #2e4256; border-radius: 12px;
    }
    [data-testid="stSidebar"] { background: #0b1019; border-right: 1px solid #1c2839; }
</style>
"""

MODE_TEXT = "✏️  Text watermark"
MODE_LOGO = "🏷️  Logo watermark"
MODE_OPTIONS = (MODE_TEXT, MODE_LOGO)


# --------------------------------------------------------------------------- #
# Small render helpers
# --------------------------------------------------------------------------- #

def metric_row(cards: list[tuple[str, str, str]]) -> None:
    """Render metric cards (icon, label, value) as one responsive HTML row."""
    body = "".join(
        f'<div class="dscard"><div class="dsicon">{icon}</div>'
        f'<div class="dslabel">{label}</div><div class="dsvalue">{value}</div></div>'
        for icon, label, value in cards
    )
    st.markdown(f'<div class="dscard-row">{body}</div>', unsafe_allow_html=True)


def preview_image(img: Image.Image, max_side: int = 1200) -> Image.Image:
    """Shrink + flatten RGBA onto white — cheap, safe copy for st.image."""
    copy = img.copy()
    copy.thumbnail((max_side, max_side))
    if copy.mode == "RGBA":
        bg = Image.new("RGBA", copy.size, (255, 255, 255, 255))
        copy = Image.alpha_composite(bg, copy)
    return copy.convert("RGB")


def safe_stem(filename: str) -> str:
    return re.sub(r"[^\w\- ]+", "_", os.path.splitext(filename or "image")[0]).strip() or "image"


# --------------------------------------------------------------------------- #
# Header / empty state
# --------------------------------------------------------------------------- #

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <div class="hero-emoji">🖼️</div>
        <div>
            <div class="hero-title">Watermark Studio</div>
            <div class="hero-sub">Upload an image → add a text or logo watermark →
            preview side-by-side → download a pristine PNG. The original file is never touched.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "📤  **1 · Upload your image**  (PNG · JPG · WEBP · BMP · TIFF · GIF)",
    type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "gif"],
)

# --------------------------------------------------------------------------- #
# Sidebar: brand + watermark controls (only meaningful once an image exists)
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-emoji">💧</div>
            <div>
                <div class="brand-name">Watermark Studio</div>
                <div class="brand-tag">non-destructive · PNG out</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if uploaded is None:
        st.info("Upload an image on the left to reveal the watermark controls.", icon="🎛️")

if uploaded is None:
    st.markdown(
        """
        <div class="empty-card">
            <div class="empty-emoji">🛡️</div>
            <div class="empty-title">Protect your pictures in seconds</div>
            <div class="empty-sub">Watermark Studio stamps your images with a crisp text mark or
            your own logo — fully adjustable, fully reversible (the upload is never modified).</div>
            <div class="step-grid">
                <div class="step-card"><div class="step-num">STEP 1</div>
                <div class="step-txt">📤 Upload any image up to a few thousand pixels per side.</div></div>
                <div class="step-card"><div class="step-num">STEP 2</div>
                <div class="step-txt">🎛️ Pick text or logo, then tune size, opacity, position &amp; colour.</div></div>
                <div class="step-card"><div class="step-num">STEP 3</div>
                <div class="step-txt">⬇️ Compare the result side-by-side and download the PNG.</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Decode upload (must not crash on transparent PNGs / odd formats)
# --------------------------------------------------------------------------- #

try:
    prepared = wm.prepare_image(uploaded.getvalue())
except Exception as exc:  # corrupt / truncated file
    st.error(f"Could not read that image file: {exc}")
    st.stop()

stem = safe_stem(uploaded.name)
width, height = prepared.size

# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown('<div class="section-label">Watermark type</div>', unsafe_allow_html=True)
    mode = st.radio("Watermark type", MODE_OPTIONS, key="wm_mode", label_visibility="collapsed")

    params: dict = {}

    if mode == MODE_TEXT:
        st.markdown('<div class="section-label">✏️ Text watermark</div>', unsafe_allow_html=True)
        params["text"] = st.text_input(
            "Watermark text", value="© 2026", key="wt_text",
            help="Single-line text, e.g. © 2026 · Your Studio",
        )
        default_font = max(16, min(200, width // 12))
        params["font_size"] = st.slider(
            "Font size (px)", 16, 300, int(default_font), step=2, key="wt_font",
        )
        params["color"] = st.color_picker("Watermark colour", "#ffffff", key="wt_color")

        c1, c2 = st.columns(2)
        params["opacity"] = c1.slider("Opacity (%)", 1, 100, 60, key="wt_opacity") / 100.0
        params["angle"] = c2.slider("Rotation (°)", -90, 90, 0, step=1, key="wt_angle")
        params["outline"] = st.checkbox(
            "Dark contrast outline", value=True, key="wt_outline",
            help="Adds a subtle dark edge so light text stays readable on bright areas.",
        )

        params["tile"] = st.checkbox(
            "Tile across the whole image", value=False, key="wt_tile",
            help="Repeat the text in a grid instead of placing it once.",
        )
        params["position"] = "Bottom Right"
        params["x_percent"], params["y_percent"] = 50.0, 50.0
        if params["tile"]:
            params["gap_percent"] = st.slider(
                "Gap between tiles (% of text width)", 0, 300, 40, key="wt_gap",
            )
        else:
            c3, c4 = st.columns(2)
            params["position"] = c3.selectbox(
                "Position", list(wm.POSITIONS), index=8, key="wt_pos",
            )
            st.caption("Position applies to the single text placement.")
            if params["position"] == "Custom":
                params["x_percent"] = st.slider("Across (%)", 0, 100, 80, key="wt_x")
                params["y_percent"] = st.slider("Down (%)", 0, 100, 10, key="wt_y")
    else:  # MODE_LOGO
        st.markdown('<div class="section-label">🏷️ Logo watermark</div>', unsafe_allow_html=True)
        logo_file = st.file_uploader(
            "Upload your logo (PNG with transparency works best)",
            type=["png", "jpg", "jpeg", "webp"], key="lo_file",
        )
        logo: Image.Image | None = None
        if logo_file is not None:
            try:
                logo = wm.prepare_image(logo_file.getvalue()).rgba
            except Exception as exc:
                st.error(f"Could not read the logo file: {exc}")
        if logo is not None:
            thumb = preview_image(logo, max_side=140)
            c, _ = st.columns([1, 2])
            c.image(thumb, caption=f"Logo · {logo.width}×{logo.height} px", width=120)

        params["width_percent"] = st.slider(
            "Logo width (% of image width)", 3, 100, 18, step=1, key="lo_width",
        )
        params["opacity"] = st.slider("Opacity (%)", 1, 100, 70, key="lo_opacity") / 100.0
        params["position"] = st.selectbox(
            "Position", list(wm.POSITIONS), index=8, key="lo_pos",
        )
        if params["position"] == "Custom":
            params["x_percent"] = st.slider("Across (%)", 0, 100, 80, key="lo_x")
            params["y_percent"] = st.slider("Down (%)", 0, 100, 80, key="lo_y")

# --------------------------------------------------------------------------- #
# Guards & processing
# --------------------------------------------------------------------------- #

if mode == MODE_LOGO and logo_file is None:
    st.markdown(
        '<div class="warn-card">⚠️ &nbsp;Please upload a <b>logo image</b> in the sidebar '
        "to continue — or switch back to the text watermark.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

if mode == MODE_TEXT and not (params.get("text") or "").strip():
    st.markdown(
        '<div class="warn-card">⚠️ &nbsp;Please type some watermark text in the sidebar.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

try:
    if mode == MODE_TEXT:
        result = wm.apply_text_watermark(prepared.rgba, **params)
        mode_short = "Text"
    else:
        result = wm.apply_logo_watermark(prepared.rgba, logo, **params)
        mode_short = "Logo"
except (ValueError, OSError) as exc:
    st.error(f"Could not apply the watermark: {exc}")
    result = prepared.rgba.copy()

# --------------------------------------------------------------------------- #
# Metrics + side-by-side preview + download
# --------------------------------------------------------------------------- #

png_bytes = wm.encode_png(result, keep_alpha=prepared.has_alpha)
result_rgb = result.convert("RGB")  # flattened copy used only for display

metric_row(
    [
        ("📐", "Original", f"{width} × {height} px"),
        ("🗂️", "Source format", prepared.source_format),
        ("🧰", "Watermark", mode_short),
        ("📦", "Download size", f"{len(png_bytes) / 1024:.0f} KB"),
    ]
)

col_a, col_b = st.columns(2, gap="large")
with col_a:
    st.markdown('<div class="shot-title">🔹 Original — untouched</div>', unsafe_allow_html=True)
    st.image(preview_image(prepared.rgba), use_container_width=True)
with col_b:
    st.markdown('<div class="shot-title">💧 Watermarked result</div>', unsafe_allow_html=True)
    st.image(preview_image(result), use_container_width=True)

st.markdown("---")

left, mid, right = st.columns([1, 2, 1])
with mid:
    st.download_button(
        "⬇️  Download watermarked PNG",
        data=png_bytes,
        file_name=f"{stem}_watermarked.png",
        mime="image/png",
        type="primary",
        use_container_width=True,
    )
    st.caption(
        f"Lossless PNG · {width}×{height} px · alpha "
        f"{'preserved' if prepared.has_alpha else 'flattened to RGB'}."
    )

st.markdown(
    '<div class="footer">Watermark Studio · Pillow compositing on a copy — your upload '
    "is never modified or stored · Powered by Streamlit</div>",
    unsafe_allow_html=True,
)
