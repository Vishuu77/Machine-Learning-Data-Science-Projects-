# -*- coding: utf-8 -*-
"""
app.py
======
German Traffic Sign Classification — Streamlit app.

Loads the artifacts produced by train_model.py (model.pt + meta.json),
lets the user upload a photo of a German traffic sign (or pick one of the
bundled demo samples) and shows the TOP-3 predictions with confidence bars
plus the meaning of the sign.

Run:
    python -m streamlit run app.py --server.port 2002
"""
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from sign_info import CLASS_INFO

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.pt"
META_PATH = HERE / "meta.json"
SAMPLES_DIR = HERE / "samples"


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the TorchScript model + metadata once per session."""
    model = torch.jit.load(str(MODEL_PATH), map_location="cpu")
    model.eval()
    meta = json.loads(META_PATH.read_text("utf-8"))
    return model, meta


def preprocess(img: Image.Image, size: int) -> torch.Tensor:
    """Same preprocessing as training: RGB -> resize -> float [0,1] -> NCHW."""
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def predict(model, meta, img: Image.Image):
    t0 = time.perf_counter()
    with torch.inference_mode():
        probs = model(preprocess(img, int(meta["img_size"])))[0].numpy()
    ms = (time.perf_counter() - t0) * 1000.0
    top_idx = np.argsort(probs)[::-1][:3]
    top3 = []
    for rank, c in enumerate(top_idx, start=1):
        c = int(c)
        top3.append({
            "rank": rank,
            "class_id": c,
            "german": meta["german"].get(str(c), CLASS_INFO.get(c, {}).get("de", "?")),
            "english": meta["class_labels"].get(str(c), CLASS_INFO.get(c, {}).get("en", "?")),
            "meaning": meta["meaning"].get(str(c), CLASS_INFO.get(c, {}).get("note", "")),
            "conf": float(probs[c]) * 100.0,
        })
    return top3, ms


def local_css():
    st.markdown(
        """
        <style>
        .stApp { background: radial-gradient(1200px 600px at 15% -5%, #14203c 0%, #0b0f19 55%); }
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
        .hero-title {
            font-size: 2.35rem; font-weight: 800; letter-spacing: -.5px;
            background: linear-gradient(90deg, #ffd54f, #ff9800, #ffd54f);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .hero-sub { color: #9fb0cf; margin-top: -6px; font-size: .98rem; }
        .pred-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0; }
        .pred-german { color: #ffd54f; font-size: 1.7rem; font-weight: 800; line-height: 1.2; }
        .pred-english { color: #cdd9f0; font-size: 1.02rem; }
        .chip {
            display: inline-block; background: #17213a; border: 1px solid #2b3a63;
            border-radius: 999px; padding: 3px 12px; margin: 2px 4px 2px 0;
            font-size: .8rem; color: #c7d4ee;
        }
        .small { color: #7e8db0; font-size: .82rem; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
#  page
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="German Traffic Sign Classifier",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)
local_css()

if not (MODEL_PATH.exists() and META_PATH.exists()):
    st.markdown('<p class="hero-title">🚦 German Traffic Sign Classifier</p>', unsafe_allow_html=True)
    st.warning("🚧 No trained model found yet.")
    st.markdown(
        "Run the training script first to download a public GTSRB dataset and train the CNN:\n\n"
        "```bash\ncd course_projects/02_traffic_sign\npython train_model.py\n```\n\n"
        "then restart this app. Artifacts created: `model.pt`, `meta.json`, `samples/`."
    )
    st.stop()

model, meta = load_artifacts()
classes = [int(c) for c in meta["classes"]]
val_acc = float(meta["val_acc_pct"])
img_size = int(meta["img_size"])

# ---------------- header ------------------------------------------------ #
st.markdown('<p class="hero-title">🚗 German Traffic Sign Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Upload a photo of a German road sign — a real CNN '
    f"(trained on {len(classes)} GTSRB classes) predicts its meaning.</p>",
    unsafe_allow_html=True,
)

# ---------------- metric cards ------------------------------------------- #
m1, m2, m3, m4 = st.columns(4)
m1.metric("🗂️ Trained classes", f"{len(classes)}", border=True)
m2.metric("🎯 Validation accuracy", f"{val_acc:.1f} %", border=True)
m3.metric("🖼️ Input size", f"{img_size}×{img_size} RGB", border=True)
m4.metric("📦 Model", "TorchScript CNN", border=True)

# ---------------- input row ---------------------------------------------- #
c_up, c_sel = st.columns([2.1, 1])
uploaded = c_up.file_uploader(
    "📤 Upload a traffic-sign image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)
sample_files = sorted([p.name for p in SAMPLES_DIR.glob("*.png")]) if SAMPLES_DIR.exists() else []
sample_choice = None
if sample_files:
    sample_choice = c_sel.selectbox(
        "🖼️ …or try a bundled sample", sample_files, index=0,
        help="Demo images cropped from the validation split of the dataset.",
    )

if uploaded is not None:
    src_name = uploaded.name
    img = Image.open(io.BytesIO(uploaded.getvalue()))
else:
    src_name = sample_choice
    img = Image.open(SAMPLES_DIR / sample_choice) if sample_choice else None

if img is None:
    st.info("👈 Upload an image or choose a sample to get started.")
    st.stop()

img = img.convert("RGB")

# ---------------- image + results ---------------------------------------- #
col_img, col_res = st.columns([1, 1.35], gap="large")

with col_img:
    with st.container(border=True):
        st.markdown("**📷 Input image**")
        st.image(img, width="stretch")
        st.markdown(f'<span class="small">file: `{src_name}` · original {img.size[0]}×{img.size[1]} px</span>',
                    unsafe_allow_html=True)

top3, infer_ms = predict(model, meta, img)
best = top3[0]

with col_res:
    with st.container(border=True):
        st.markdown("**🔎 Prediction**")
        st.markdown(
            f'<p class="pred-german">{best["german"]}</p>'
            f'<p class="pred-english">{best["english"]} · '
            f'<span class="chip">class {best["class_id"]}</span>'
            f'<span class="chip">conf {best["conf"]:.1f} %</span></p>',
            unsafe_allow_html=True,
        )
        st.progress(min(best["conf"] / 100.0, 1.0))
        st.markdown(
            f'<p class="small">💡 {best["meaning"]}  ·  inference {infer_ms:.1f} ms on CPU</p>',
            unsafe_allow_html=True,
        )

# ---------------- top-3 chart + table ------------------------------------- #
st.markdown("### 📊 Top-3 predictions")
chart_df = pd.DataFrame(
    {"confidence (%)": [t["conf"] for t in top3]},
    index=[f"{t['german']} ({t['conf']:.1f}%)" for t in top3],
)
st.bar_chart(chart_df, height=280, color="#ffb300")

table_df = pd.DataFrame(
    [
        {
            "Rank": t["rank"],
            "Class": t["class_id"],
            "German sign": t["german"],
            "English": t["english"],
            "Meaning": t["meaning"],
            "Confidence": f"{t['conf']:.1f} %",
        }
        for t in top3
    ]
)
st.dataframe(table_df, hide_index=True, width="stretch")

# ---------------- expandable details --------------------------------------- #
with st.expander("🧠 How does it work?"):
    st.markdown(
        f"- **Model:** tiny 3-layer CNN (conv → BN → ReLU → maxpool ×3 + dropout MLP), "
        f"trained from scratch on **CPU** with PyTorch.\n"
        f"- **Preprocessing:** convert to RGB → resize to {img_size}×{img_size} → scale to [0,1] "
        f"(identical to training).\n"
        f"- **Classes trained:** {len(classes)} German signs (GTSRB ids {classes}).\n"
        f"- **Dataset:** `{meta['dataset']}` · {meta['dataset_url']}\n"
        f"- **Validation accuracy:** {val_acc:.2f} % on a held-out split "
        f"({meta.get('n_val', '?')} images) · trained {meta.get('epochs_run', '?')} epochs.\n"
        "- **Note:** the CNN works best on **cropped, close-up sign photos** — the model was "
        "trained on tight crops of the sign, not full street scenes."
    )

with st.expander("📚 Class dictionary (German sign meanings)"):
    rows = []
    for c in classes:
        rows.append(
            f'- **{meta["german"].get(str(c), "?")}** — {meta["class_labels"].get(str(c), "")} '
            f'<span class="chip">GTSRB class {c}</span> — {meta["meaning"].get(str(c), "")}'
        )
    st.markdown("\n".join(rows), unsafe_allow_html=True)

# ---------------- sidebar -------------------------------------------------- #
with st.sidebar:
    st.markdown("## 🚦 About")
    st.markdown(
        "Classifies **German traffic signs** from the classic **GTSRB** benchmark "
        "into one of 12 trained classes, showing the **top-3** guesses with confidence."
    )
    st.markdown("#### 🔁 Retrain")
    st.markdown(
        "```bash\ncd course_projects/02_traffic_sign\npython train_model.py\n```"
    )
    st.markdown("#### 🎯 Trained classes")
    st.markdown(
        ", ".join(f"**{c}**" for c in classes)
        + "  (GTSRB ids — see the class dictionary above for names)"
    )
    st.markdown(
        "#### 📚 Data\n"
        "[tanganke/gtsrb](https://huggingface.co/datasets/tanganke/gtsrb) on Hugging Face — "
        "GTSRB (J. Stallkamp et al., IJCNN 2011) re-hosted, no auth required."
    )
    st.markdown("---")
    st.caption("Course project · Streamlit + PyTorch · CPU only")
