"""🎭 Face Swap Studio — Streamlit app (port 2006, CPU-only).

Upload a SOURCE face photo and a TARGET person photo; the app pastes the
source person's face onto the target photo and lets you download the result.
Processing lives in face_swap_engine.py (no streamlit imports there).
"""
import hashlib
import os

import cv2
import numpy as np
import streamlit as st

import face_swap_engine as fse

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_SRC = os.path.join(ROOT, "samples", "source_man.jpg")
SAMPLE_TGT = os.path.join(ROOT, "samples", "target_woman.jpg")

st.set_page_config(page_title="Face Swap Studio", page_icon="🎭", layout="wide")

st.markdown("""<style>
    .stApp { background: #0d1117; }
    html, body, [class*="css"] { font-family: 'Segoe UI', Roboto, sans-serif; }
    .title { text-align:center; font-size:2.6rem; font-weight:800; margin:.2rem 0 .1rem 0;
             background: linear-gradient(90deg,#ff6ec7,#7873f5,#4ade80);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .subtitle { text-align:center; color:#8b949e; margin-bottom:1.2rem; }
    .panel { background:#161b22; border:1px solid #30363d; border-radius:14px; padding:1rem; margin:.4rem 0; }
    .card { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:.8rem 1rem; text-align:center; }
    .card .k { color:#8b949e; font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; }
    .card .v { color:#e6edf3; font-size:1.15rem; font-weight:700; margin-top:.2rem; }
    .colcap { text-align:center; font-weight:700; font-size:1rem; padding:.25rem 0 .1rem 0; }
    .colcap.src { color:#58a6ff; } .colcap.tgt { color:#ffa657; } .colcap.out { color:#4ade80; }
    .stButton>button { border-radius:10px; font-weight:600; }
    footer {visibility:hidden;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="title">🎭 Face Swap Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Paste the <b>source face</b> onto the <b>target photo</b> — real photos, frontal faces, CPU-only</div>',
            unsafe_allow_html=True)

# ---------------- sidebar ----------------------------------------------------
with st.sidebar:
    st.markdown("### 🧰 Controls")
    use_samples = st.checkbox("✨ Use bundled sample photos", value=True,
                              help="No upload needed — swaps a sample man's face onto a sample woman's photo.")
    st.markdown("---")
    st.markdown("#### 📤 SOURCE face")
    up_src = None if use_samples else st.file_uploader("Upload the person whose face will appear", type=["jpg", "jpeg", "png"],
                                                       key="src_up", label_visibility="collapsed")
    st.markdown("#### 🎯 TARGET photo")
    up_tgt = None if use_samples else st.file_uploader("Upload the person who will receive that face", type=["jpg", "jpeg", "png"],
                                                       key="tgt_up", label_visibility="collapsed")
    st.markdown("---")
    st.markdown("#### ⚙️ Engine status")
    mp_st = fse.mediapipe_status()
    eng = "✅ mediapipe v" + str(mp_st["version"]) if mp_st["available"] else "⚠️ OpenCV Haar fallback"
    st.markdown(f"<div class='panel' style='padding:.5rem .7rem; font-size:.85rem; color:#e6edf3;'>{eng}"
                f"<br><span style='color:#8b949e'>model file: {'✓' if mp_st['model_exists'] else '✗'} "
                f"face_landmarker.task</span></div>", unsafe_allow_html=True)
    st.markdown("#### 💡 Tips")
    st.caption("Front-facing, well-lit photos work best. One clear face per image. "
               "Results are demo-grade — skin tone is auto-matched, pose comes from the target.")

def read_upload(uploaded):
    if uploaded is None:
        return None
    return uploaded.getvalue()

def read_sample(path):
    with open(path, "rb") as f:
        return f.read()

src_bytes = None
tgt_bytes = None
if use_samples:
    src_bytes, tgt_bytes = read_sample(SAMPLE_SRC), read_sample(SAMPLE_TGT)
    src_name, tgt_name = "sample: source_man.jpg", "sample: target_woman.jpg"
else:
    src_bytes, tgt_bytes = read_upload(up_src), read_upload(up_tgt)
    src_name = up_src.name if up_src else None
    tgt_name = up_tgt.name if up_tgt else None

# ---------------- inputs preview ---------------------------------------------
st.markdown("### 1️⃣ &nbsp;Inputs")
if src_bytes is None or tgt_bytes is None:
    st.info("👈 Choose **sample photos** in the sidebar, or upload a SOURCE face and a TARGET photo to get started.")
else:
    col_s, col_t = st.columns(2)
    with col_s:
        s_bgr = fse.decode_image(src_bytes)
        st.image(cv2.cvtColor(s_bgr, cv2.COLOR_BGR2RGB), width="stretch")
        st.markdown(f"<div class='colcap src'>📤 SOURCE FACE — {src_name}</div>", unsafe_allow_html=True)
    with col_t:
        t_bgr = fse.decode_image(tgt_bytes)
        st.image(cv2.cvtColor(t_bgr, cv2.COLOR_BGR2RGB), width="stretch")
        st.markdown(f"<div class='colcap tgt'>🎯 TARGET PHOTO — {tgt_name}</div>", unsafe_allow_html=True)

# ---------------- swap ---------------------------------------------------------
st.markdown("### 2️⃣ &nbsp;Swap")
swap_clicked = st.button("🪄 Swap the faces", type="primary", width="stretch")

sig = None
if src_bytes is not None and tgt_bytes is not None:
    sig = hashlib.md5(src_bytes).hexdigest()[:10] + "|" + hashlib.md5(tgt_bytes).hexdigest()[:10]

RES_KEY, INFO_KEY, ERR_KEY = "res", "info", "err"
if swap_clicked or (sig and st.session_state.get("sig") != sig):
    if src_bytes is not None and tgt_bytes is not None:
        with st.spinner("Detecting faces, warping landmarks, blending…"):
            try:
                result_bgr, info = fse.swap_faces(src_bytes, tgt_bytes)
                st.session_state[RES_KEY] = result_bgr
                st.session_state[INFO_KEY] = info
                st.session_state.pop(ERR_KEY, None)
                st.session_state["sig"] = sig
            except Exception as exc:
                st.session_state[ERR_KEY] = str(exc)

if ERR_KEY in st.session_state:
    st.error("😕 Swap failed: " + st.session_state[ERR_KEY])
    st.caption("Make sure each photo contains one clear, front-facing face.")
elif RES_KEY in st.session_state and st.session_state.get("sig") == sig:
    result_bgr = st.session_state[RES_KEY]
    info = st.session_state[INFO_KEY]

    m1, m2, m3, m4 = st.columns(4)
    cards = [
        ("Method", info["method"], "🧠"),
        ("Swap time", f"{info['time_ms']} ms", "⏱️"),
        ("Input", f"{info['src_shape'][0]}×{info['src_shape'][1]}", "🖼️"),
        ("Output", f"{info['out_shape'][0]}×{info['out_shape'][1]}", "📐"),
    ]
    for col, (k, v, ic) in zip((m1, m2, m3, m4), cards):
        with col:
            st.markdown(f"<div class='card'><div class='k'>{ic} {k}</div><div class='v'>{v}</div></div>",
                        unsafe_allow_html=True)

    st.markdown("### 3️⃣ &nbsp;Result")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.image(cv2.cvtColor(fse.decode_image(src_bytes), cv2.COLOR_BGR2RGB), width="stretch")
        st.markdown("<div class='colcap src'>📤 SOURCE</div>", unsafe_allow_html=True)
    with col_b:
        st.image(cv2.cvtColor(fse.decode_image(tgt_bytes), cv2.COLOR_BGR2RGB), width="stretch")
        st.markdown("<div class='colcap tgt'>🎯 TARGET</div>", unsafe_allow_html=True)
    with col_c:
        st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), width="stretch")
        st.markdown("<div class='colcap out'>✨ RESULT</div>", unsafe_allow_html=True)

    png_bytes = fse.to_png_bytes(result_bgr)
    st.download_button("⬇️ Download result (PNG)", data=png_bytes,
                       file_name="face_swapped.png", mime="image/png",
                       width="stretch")
else:
    st.caption("Press **🪄 Swap the faces** to run (auto-runs for sample photos / new uploads).")

st.markdown("""<div style='text-align:center;color:#484f58;font-size:.8rem;margin-top:1.2rem;'>
Face Swap Studio · mediapipe FaceLandmarker + Delaunay warp · OpenCV fallback · runs 100% locally on CPU</div>""",
            unsafe_allow_html=True)
