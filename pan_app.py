import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import imutils

st.set_page_config(
    page_title="PAN Card Tampering Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; background-color: #2563eb; color: white; font-weight: bold; border-radius: 8px; }
    .stButton>button:hover { background-color: #1d4ed8; }
    .metric-card { background-color: #1f2937; padding: 20px; border-radius: 10px; border: 1px solid #374151; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🔍 PAN Tampering AI")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("Navigation", ["🔍 Tampering Detector", "📖 About & Instructions"])
st.sidebar.markdown("---")
st.sidebar.info("🛡️ **Detector Engine:** OpenCV + SSIM\n\n🎯 **Status:** Active & Ready")

if app_mode == "🔍 Tampering Detector":
    st.title("🛡️ PAN Card Tampering & Fraud Detection App")
    st.markdown("Upload a reference (original) PAN card image and a target (uploaded) PAN card image to detect structural tampering and differences.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Reference (Original) PAN Card")
        ref_file = st.file_uploader("Upload Original PAN Image", type=["jpg", "jpeg", "png"], key="ref")
        
    with col2:
        st.subheader("2. Target (Uploaded) PAN Card")
        target_file = st.file_uploader("Upload Target PAN Image", type=["jpg", "jpeg", "png"], key="target")
        
    if ref_file is not None and target_file is not None:
        # Load images
        image_original = Image.open(ref_file).convert('RGB')
        image_target = Image.open(target_file).convert('RGB')
        
        # Resize to standard dimensions for comparison (e.g., 300x200)
        width, height = 300, 200
        img_orig_resized = image_original.resize((width, height))
        img_target_resized = image_target.resize((width, height))
        
        # Convert PIL to OpenCV numpy arrays
        np_orig = np.array(img_orig_resized)
        np_target = np.array(img_target_resized)
        
        # Convert to grayscale
        gray_orig = cv2.cvtColor(np_orig, cv2.COLOR_RGB2GRAY)
        gray_target = cv2.cvtColor(np_target, cv2.COLOR_RGB2GRAY)
        
        # Compute Structural Similarity Index (SSIM)
        score, diff = ssim(gray_orig, gray_target, full=True)
        diff = (diff * 255).astype("np.uint8" if hasattr(np, 'uint8') else "uint8")
        
        # Threshold difference
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        # Draw bounding boxes on target image
        marked_target = np_target.copy()
        for c in cnts:
            if cv2.contourArea(c) > 10:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(marked_target, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="Similarity Score", value=f"{score * 100:.2f}%")
        with m2:
            status = "Authentic / Genuine ✅" if score > 0.85 else "Tampered / Modified ❌"
            st.metric(label="Verification Status", value=status)
        with m3:
            st.metric(label="Differences Detected", value=f"{len(cnts)} regions")
            
        st.markdown("---")
        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            st.image(image_original, caption="Original PAN Card", use_column_width=True)
        with res_c2:
            st.image(image_target, caption="Uploaded PAN Card", use_column_width=True)
        with res_c3:
            st.image(marked_target, caption="Tampered Areas Highlighted (Red Boxes)", use_column_width=True)
            
        if score > 0.85:
            st.success("🟢 **Result:** The uploaded PAN card structurally matches the original reference card. No tampering detected.")
        else:
            st.warning("🔴 **Result:** Significant structural deviations or tampering detected between the original and uploaded PAN card images.")
            
elif app_mode == "📖 About & Instructions":
    st.title("📖 About PAN Card Tampering Detector")
    st.markdown("""
    ### How It Works:
    1. **Structural Similarity Index (SSIM):** Compares the pixel structure, layout, fonts, and alignment between the reference PAN card and the user-uploaded PAN card.
    2. **Contour Detection & Difference Mapping:** Identifies exact coordinate regions where edits, modifications, or replacements (such as name, PAN number, or photo swapping) were made.
    3. **Automated Verdict:** Outputs a percentage similarity score and flags tampered areas with red bounding boxes.
    """)
