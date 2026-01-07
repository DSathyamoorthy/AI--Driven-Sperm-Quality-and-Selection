import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import math
import time
import pandas as pd
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# Modern Page Configuration
# -----------------------------
st.set_page_config(
    layout="wide", 
    page_title="AI Sperm Analysis Platform",
    page_icon="🔬",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6C5CE7;
        --secondary-color: #00B894;
        --accent-color: #FDCB6E;
        --danger-color: #D63031;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-title {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2D3436;
    }
    
    .metric-label {
        color: #636E72;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .status-excellent {
        background: #00B894;
        color: white;
    }
    
    .status-good {
        background: #55EFC4;
        color: #2D3436;
    }
    
    .status-fair {
        background: #FDCB6E;
        color: #2D3436;
    }
    
    .status-poor {
        background: #D63031;
        color: white;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2D3436;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #6C5CE7;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #6C5CE7;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Modern Header
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🔬 AI-Powered Sperm Analysis Platform</h1>
    <p class="main-subtitle">Advanced Computer Vision for Clinical Fertility Assessment</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Stylish Sidebar
# -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/microscope.png", width=80)
    st.markdown("### ⚙️ Configuration Panel")
    st.markdown("---")
    
    with st.expander("🤖 Model Settings", expanded=True):
        motility_path = st.text_input("Motility Model", value="models/motility_best.pt")
        morphology_path = st.text_input("Morphology Model", value="models/best.pt")
    
    with st.expander("⚡ Processing Options", expanded=True):
        frame_step = st.number_input("Frame Sampling Rate", min_value=1, value=3, step=1, help="Process every Nth frame")
        resize_width = st.number_input("Processing Width (px)", min_value=160, value=640, step=16)
        process_imgsz = st.number_input("Detection Image Size", min_value=128, max_value=1280, value=640, step=32)
        imgsz_morph = st.number_input("Morphology Crop Size", min_value=64, max_value=512, value=128, step=16)
        conf_thr = st.slider("Confidence Threshold", 0.01, 0.99, 0.20, 0.01)
    
    with st.expander("🎯 Classification Thresholds", expanded=False):
        immotile_thr = st.number_input("Immotile ≤ (px/sec)", value=3.0)
        nonprog_thr = st.number_input("Non-Progressive ≤ (px/sec)", value=15.0)
    
    with st.expander("📊 SQI Weights", expanded=False):
        w_mot = st.slider("Motility Weight", 0.0, 1.0, 0.6, 0.05)
        w_morph = st.slider("Morphology Weight", 0.0, 1.0, 0.4, 0.05)
    
    st.markdown("---")
    st.markdown("**Model Status**")
    if os.path.exists(motility_path):
        st.success("✅ Motility model loaded")
    else:
        st.error("❌ Motility model not found")
    if os.path.exists(morphology_path):
        st.success("✅ Morphology model loaded")
    else:
        st.error("❌ Morphology model not found")

# Validation
if not os.path.exists(motility_path) or not os.path.exists(morphology_path):
    st.error("⚠️ Please ensure both model files are available before proceeding.")
    st.stop()

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models(mot_p, morph_p):
    mot = YOLO(mot_p)
    morph = YOLO(morph_p)
    return mot, morph

with st.spinner("🔄 Initializing AI models..."):
    mot_model, morph_model = load_models(motility_path, morphology_path)

# Map morphology names
if isinstance(morph_model.names, dict):
    morph_names_map = {int(k): v.lower() for k, v in morph_model.names.items()}
else:
    morph_names_map = {i: v.lower() for i, v in enumerate(morph_model.names)}

# -----------------------------
# Helper Functions
# -----------------------------
def clamp(x, a, b):
    return max(a, min(b, int(x)))

class SimpleTracker:
    def __init__(self, max_dist=80, max_inactive=8):
        self.tracks = {}
        self.next_id = 0
        self.max_dist = max_dist
        self.max_inactive = max_inactive

    def update(self, detections, frame_idx):
        det_centroids = [(d['cx'], d['cy']) for d in detections]

        if not self.tracks:
            for d in detections:
                self.tracks[self.next_id] = {
                    'centroids': [(d['cx'], d['cy'])],
                    'bboxes': [d['bbox']],
                    'last_frame': frame_idx,
                    'inactive': 0,
                    'morph_checked': False,
                    'morph_label': None
                }
                self.next_id += 1
            return

        assigned = set()
        used_tracks = set()
        for i, (cx, cy) in enumerate(det_centroids):
            best_tid = None
            best_dist = None
            for tid, t in self.tracks.items():
                if tid in used_tracks:
                    continue
                tx, ty = t['centroids'][-1]
                dist = math.hypot(cx-tx, cy-ty)
                if dist <= self.max_dist and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_tid = tid
            if best_tid is not None:
                self.tracks[best_tid]['centroids'].append((cx, cy))
                self.tracks[best_tid]['bboxes'].append(detections[i]['bbox'])
                self.tracks[best_tid]['last_frame'] = frame_idx
                self.tracks[best_tid]['inactive'] = 0
                used_tracks.add(best_tid)
                assigned.add(i)
            else:
                self.tracks[self.next_id] = {
                    'centroids': [(cx, cy)],
                    'bboxes': [detections[i]['bbox']],
                    'last_frame': frame_idx,
                    'inactive': 0,
                    'morph_checked': False,
                    'morph_label': None
                }
                self.next_id += 1
                assigned.add(i)

        to_delete = []
        for tid, t in list(self.tracks.items()):
            if t['last_frame'] < frame_idx:
                t['inactive'] += 1
            if t['inactive'] > self.max_inactive:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

    def summarize(self, fps=None):
        out = []
        for tid, t in self.tracks.items():
            if len(t['centroids']) >= 2:
                dists = [math.hypot(t['centroids'][i][0]-t['centroids'][i-1][0],
                                    t['centroids'][i][1]-t['centroids'][i-1][1]) 
                         for i in range(1, len(t['centroids']))]
                avg_px_per_frame = sum(dists)/len(dists)
            else:
                avg_px_per_frame = 0.0
            avg_px_per_sec = avg_px_per_frame * (fps or 1.0)
            out.append({
                'track_id': tid,
                'avg_px_per_frame': avg_px_per_frame,
                'avg_px_per_sec': avg_px_per_sec,
                'morph_label': t.get('morph_label', None),
                'frames_seen': len(t['centroids'])
            })
        return out

# -----------------------------
# Upload Section
# -----------------------------
st.markdown('<p class="section-header">📁 Upload Sample</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    uploaded = st.file_uploader(
        "Select sperm analysis video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a microscopy video for automated analysis"
    )

if not uploaded:
    st.markdown("""
    <div class="info-box">
        <h3>📋 Getting Started</h3>
        <p>Upload a microscopy video of sperm sample to begin automated analysis. The system will:</p>
        <ul>
            <li>🎯 Detect and track individual sperm cells</li>
            <li>🏃 Classify motility patterns (WHO standards)</li>
            <li>🔬 Analyze morphological characteristics</li>
            <li>📊 Generate comprehensive quality metrics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Save to temp
tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
tmp_in.write(uploaded.read())
tmp_in.flush()
tmp_in.close()
video_path = tmp_in.name

# Video info
cap = cv2.VideoCapture(video_path)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

# Display video info in modern cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📐 Resolution", f"{orig_w}×{orig_h}")
with col2:
    st.metric("🎬 FPS", f"{fps:.1f}")
with col3:
    st.metric("🎞️ Total Frames", f"{frame_count}")
with col4:
    duration = frame_count / fps if fps > 0 else 0
    st.metric("⏱️ Duration", f"{duration:.1f}s")

with st.expander("🎥 Video Preview", expanded=True):
    st.video(video_path)

# -----------------------------
# Processing
# -----------------------------
if st.button("🚀 Start Analysis", use_container_width=True):
    # Extract frames
    frames = []
    frame_idxs = []
    idx = 0
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    status_text.text("📥 Extracting frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % frame_step != 0:
            continue
        h0, w0 = frame.shape[:2]
        if w0 != resize_width:
            scale = resize_width / float(w0)
            frame_proc = cv2.resize(frame, (resize_width, int(h0*scale)))
        else:
            frame_proc = frame
        frames.append(frame_proc)
        frame_idxs.append(idx)
        if frame_count:
            progress_bar.progress(min(1.0, idx/frame_count))
    
    cap.release()
    progress_bar.empty()
    
    if len(frames) == 0:
        st.error("❌ No frames extracted. Please adjust frame sampling rate.")
        st.stop()
    
    status_text.text(f"✅ Extracted {len(frames)} frames")
    
    # Prepare output
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = out_tmp.name
    h_out, w_out = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(out_path, fourcc, fps/(frame_step or 1), (w_out, h_out))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps/(frame_step or 1), (w_out, h_out))
    
    tracker = SimpleTracker(max_dist=80, max_inactive=6)
    annot_preview = []
    
    # Main processing
    progress_bar = st.progress(0.0)
    status_text.text("🔄 Running AI analysis...")
    
    for fi, frame in enumerate(frames):
        try:
            mot_res = mot_model.predict(frame, imgsz=process_imgsz, conf=conf_thr, verbose=False)[0]
        except Exception:
            mot_res = mot_model(frame)[0]
        
        mot_detections = []
        if hasattr(mot_res, "boxes") and mot_res.boxes is not None and len(mot_res.boxes):
            arr = mot_res.boxes.xyxy.cpu().numpy()
            confs = mot_res.boxes.conf.cpu().numpy() if hasattr(mot_res.boxes, "conf") else np.ones(len(arr))
            for i, b in enumerate(arr):
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                mot_detections.append({"bbox":(x1,y1,x2,y2), "cx":cx, "cy":cy, "conf":float(confs[i])})
        
        tracker.update(mot_detections, frame_idx=fi)
        
        # Morphology analysis
        for tid, t in tracker.tracks.items():
            if t.get('morph_checked', False):
                continue
            if not t.get('bboxes'):
                continue
            x1, y1, x2, y2 = t['bboxes'][-1]
            H, W = frame.shape[:2]
            x1c = clamp(x1, 0, W-1)
            y1c = clamp(y1, 0, H-1)
            x2c = clamp(x2, 0, W-1)
            y2c = clamp(y2, 0, H-1)
            if x2c-x1c <= 2 or y2c-y1c <= 2:
                t['morph_checked'] = True
                t['morph_label'] = "abnormal"
                continue
            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                t['morph_checked'] = True
                t['morph_label'] = "abnormal"
                continue
            try:
                morph_res = morph_model.predict(crop, imgsz=imgsz_morph, conf=conf_thr, verbose=False)[0]
            except Exception:
                morph_res = morph_model(crop)[0]
            found_parts = set()
            if hasattr(morph_res, "boxes") and morph_res.boxes is not None and len(morph_res.boxes):
                cls_ids = morph_res.boxes.cls.cpu().numpy()
                for cid in cls_ids:
                    name = morph_names_map.get(int(cid), str(int(cid))).lower()
                    found_parts.add(name)
            needed = {"head", "neck", "tail"}
            if needed.issubset(found_parts):
                t['morph_label'] = "normal"
            else:
                t['morph_label'] = "abnormal"
            t['morph_checked'] = True
        
        # Annotate
        vis = frame.copy()
        summary = tracker.summarize(fps=fps/(frame_step or 1))
        mot_map = {}
        for s in summary:
            tid = s['track_id']
            avg_px_s = s['avg_px_per_sec']
            if avg_px_s <= immotile_thr:
                mot_lbl = "immotile"
                col = (0, 0, 255)
            elif avg_px_s <= nonprog_thr:
                mot_lbl = "non-prog"
                col = (0, 165, 255)
            else:
                mot_lbl = "progressive"
                col = (0, 255, 0)
            mot_map[tid] = (mot_lbl, col)
        
        for tid, t in tracker.tracks.items():
            cx, cy = t['centroids'][-1]
            mot_lbl, col = mot_map.get(tid, ("?", (255, 255, 255)))
            morph_lbl = t.get('morph_label', 'unknown')
            cv2.circle(vis, (int(cx), int(cy)), 4, col, -1)
            cv2.putText(vis, f"ID:{tid}", (int(cx)+8, int(cy)-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        
        if len(annot_preview) < 6:
            annot_preview.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        
        out.write(vis)
        progress_bar.progress((fi + 1) / len(frames))
    
    out.release()
    progress_bar.empty()
    status_text.text("✅ Analysis complete!")
    
    # -----------------------------
    # Results Dashboard
    # -----------------------------
    st.markdown('<p class="section-header">📊 Analysis Results</p>', unsafe_allow_html=True)
    
    tracks_summary = tracker.summarize(fps=fps/(frame_step or 1))
    for t in tracks_summary:
        avg_s = t['avg_px_per_sec']
        if avg_s <= immotile_thr:
            t['motility'] = "immotile"
        elif avg_s <= nonprog_thr:
            t['motility'] = "non-progressive"
        else:
            t['motility'] = "progressive"
        t['morphology'] = tracker.tracks[t['track_id']].get('morph_label', 'abnormal')
    
    df = pd.DataFrame(tracks_summary)
    total = len(df)
    
    if total > 0:
        mot_counts = df['motility'].value_counts().to_dict()
        morph_counts = df['morphology'].value_counts().to_dict()
        
        progressive = mot_counts.get("progressive", 0)
        normal_morph = morph_counts.get("normal", 0)
        prog_frac = progressive / total
        normal_frac = normal_morph / total
        wsum = (w_mot + w_morph) if (w_mot + w_morph) > 0 else 1.0
        sqi = 100.0 * ((w_mot/wsum)*prog_frac + (w_morph/wsum)*normal_frac)
        
        # SQI Status
        if sqi >= 75:
            status_class = "excellent"
            status_text_val = "Excellent"
        elif sqi >= 50:
            status_class = "good"
            status_text_val = "Good"
        elif sqi >= 25:
            status_class = "fair"
            status_text_val = "Fair"
        else:
            status_class = "poor"
            status_text_val = "Poor"
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sperm Quality Index</div>
                <div class="metric-value">{sqi:.1f}</div>
                <span class="status-badge status-{status_class}">{status_text_val}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Detected</div>
                <div class="metric-value">{total}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            prog_pct = (progressive/total*100) if total > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Progressive Motility</div>
                <div class="metric-value">{prog_pct:.1f}%</div>
                <small>{progressive} sperm</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            norm_pct = (normal_morph/total*100) if total > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Normal Morphology</div>
                <div class="metric-value">{norm_pct:.1f}%</div>
                <small>{normal_morph} sperm</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # Motility pie chart with Plotly
            mot_counts = df["motility"].value_counts()
            colors = {'progressive': '#00B894', 'non-progressive': '#FDCB6E', 'immotile': '#D63031'}
            fig1 = go.Figure(data=[go.Pie(
                labels=mot_counts.index,
                values=mot_counts.values,
                marker=dict(colors=[colors.get(x, '#95A5A6') for x in mot_counts.index]),
                hole=0.4
            )])
            fig1.update_layout(
                title="Motility Classification (WHO Standards)",
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Morphology pie chart
            morph_counts = df["morphology"].value_counts()
            colors_morph = {'normal': '#00B894', 'abnormal': '#D63031'}
            fig2 = go.Figure(data=[go.Pie(
                labels=morph_counts.index,
                values=morph_counts.values,
                marker=dict(colors=[colors_morph.get(x, '#95A5A6') for x in morph_counts.index]),
                hole=0.4
            )])
            fig2.update_layout(
                title="Morphology Distribution",
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Velocity distribution
        st.markdown('<p class="section-header">🏃 Velocity Analysis</p>', unsafe_allow_html=True)
        fig3 = px.histogram(
            df, 
            x="avg_px_per_sec", 
            nbins=30,
            color="motility",
            color_discrete_map={'progressive': '#00B894', 'non-progressive': '#FDCB6E', 'immotile': '#D63031'},
            title="Distribution of Sperm Velocities"
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Preview frames
        if annot_preview:
            st.markdown('<p class="section-header">🎬 Annotated Frame Samples</p>', unsafe_allow_html=True)
            cols = st.columns(3)
            for i, img in enumerate(annot_preview[:6]):
                cols[i % 3].image(img, use_container_width=True, caption=f"Frame {i+1}")
        
        # Data table
        st.markdown('<p class="section-header">📋 Detailed Track Data</p>', unsafe_allow_html=True)
        st.dataframe(
            df.style.background_gradient(subset=['avg_px_per_sec'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )
        
        # Downloads
        st.markdown('<p class="section-header">💾 Export Results</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            with open(out_path, "rb") as f:
                st.download_button(
                    "📹 Download Annotated Video",
                    f.read(),
                    file_name="annotated_output.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
        
        with col2:
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Download CSV Report",
                data=csv_bytes,
                file_name="sperm_analysis_report.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    else:
        st.warning("⚠️ No sperm cells detected in the video. Please check the video quality and model configuration.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #636E72; padding: 2rem;'>
    <p>Powered by YOLOv8 & Streamlit | AI-Driven Fertility Diagnostics</p>
</div>
""", unsafe_allow_html=True)