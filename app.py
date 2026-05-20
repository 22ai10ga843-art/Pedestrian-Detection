import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
from PIL import Image
import tempfile
import time

st.set_page_config(page_title="VisionAI | Pedestrian Detection", layout="wide", page_icon="👁️")

# ---------------- Custom CSS for God Mode UI ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700&family=Inter:wght@300;400;600&display=swap');

    /* Global App Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(0, 0, 0) 0%, rgb(15, 23, 42) 90%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Headings */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 20px rgba(0, 242, 254, 0.4);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        color: #000 !important;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.6);
    }

    /* File Uploader */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed #00f2fe;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background: rgba(0, 242, 254, 0.05);
        border-color: #4facfe;
    }

    /* Containers/Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        color: #00f2fe !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- App Header ----------------
st.markdown('<div style="text-align: center; padding-top: 2rem; padding-bottom: 2rem;">', unsafe_allow_html=True)
st.title("VISION AI: PEDESTRIAN DETECTION")
st.markdown('<p style="font-size: 1.2rem; color: #94a3b8; font-weight: 300;">Advanced HOG + SVM Computer Vision System</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Model Initialization ----------------
@st.cache_resource
def init_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog

hog = init_detector()

def detect_people(frame_bgr):
    frame = imutils.resize(frame_bgr, width=800)
    # HOG detection
    rects, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    
    # Non-maxima suppression to remove overlapping boxes
    if len(rects) > 0:
        rects_np = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        picks = non_max_suppression(rects_np, probs=None, overlapThresh=0.65)
    else:
        picks = []
        
    # Draw futuristic bounding boxes
    for (xA, yA, xB, yB) in picks:
        # Neon cyan bounding box
        cv2.rectangle(frame, (xA, yA), (xB, yB), (254, 242, 0), 2)
        # Add a sleek label
        cv2.putText(frame, "PERSON", (xA, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (254, 242, 0), 2)
        
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), len(picks)

# ---------------- Sidebar Configuration ----------------
with st.sidebar:
    st.markdown("### ⚙️ SYSTEM CONFIG")
    st.markdown("---")
    mode = st.radio("SELECT INPUT MODE", ["Image Analysis", "Video Feed"], index=0)
    st.markdown("---")
    st.markdown("### ℹ️ ABOUT")
    st.info("This system utilizes Histogram of Oriented Gradients (HOG) combined with a Support Vector Machine (SVM) to detect pedestrians in real-time.")

# ---------------- Main Interface ----------------
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📤 DATA UPLOAD")
    
    if mode == "Image Analysis":
        uploaded_img = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])
    else:
        uploaded_vid = st.file_uploader("Upload Video Stream", type=["mp4", "avi", "mov"])
        
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if mode == "Image Analysis" and uploaded_img:
        with st.spinner("Analyzing image data..."):
            # Load and convert image
            image = Image.open(uploaded_img).convert("RGB")
            bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Simulate slight processing delay for dramatic effect
            time.sleep(0.5) 
            
            # Detect
            result, count = detect_people(bgr)
            
            # Display results
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 📡 ANALYSIS RESULTS")
            
            # Metrics
            m1, m2 = st.columns(2)
            m1.metric("Subjects Detected", count)
            m2.metric("Resolution", f"{bgr.shape[1]}x{bgr.shape[0]}")
            
            st.image(result, channels="RGB", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    elif mode == "Video Feed" and 'uploaded_vid' in locals() and uploaded_vid:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📡 LIVE STREAM ANALYSIS")
        
        # Save temp video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())
        cap = cv2.VideoCapture(tfile.name)

        # UI Elements for video
        metrics_placeholder = st.empty()
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            result, count = detect_people(frame)
            
            with metrics_placeholder.container():
                m1, m2 = st.columns(2)
                m1.metric("Current Subjects", count)
                m2.metric("Status", "ACTIVE")
                
            stframe.image(result, channels="RGB", use_container_width=True)
            
        cap.release()
        st.success("Video processing complete.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Idle State
        st.markdown('<div class="glass-card" style="text-align: center; padding: 4rem 2rem;">', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #475569;">SYSTEM IDLE</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color: #64748b;">Awaiting data input in the control panel...</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
