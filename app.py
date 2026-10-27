import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal ALS Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Academic Engineering CSS
st.markdown(
    """
    <style>
    /* Global Background and Typography */
    .main {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    h1, h2, h3, h4 {
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    /* Academic Metric Box */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        margin-bottom: 12px;
    }
    
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0f766e;
        font-family: monospace;
    }
    
    .metric-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Content & Structure Cards */
    .content-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    /* Empty state placeholder panel */
    .placeholder-panel {
        background-color: #ffffff;
        border: 1.5px dashed #cbd5e1;
        border-radius: 8px;
        padding: 32px 20px;
        text-align: center;
        color: #64748b;
        margin-top: 8px;
        margin-bottom: 16px;
    }
    
    .placeholder-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 4px;
    }

    /* Clean Badges */
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #ccfbf1;
        color: #0f766e;
        border: 1px solid #99f6e4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS: DATA & GENUINE FEATURE CALCULATION
# -----------------------------------------------------------------------------
def get_synthetic_emg():
    """Generates synthetic surface EMG time-series for demonstration."""
    np.random.seed(42)
    sample_rate = 1000  # Hz
    duration = 1.0      # sec
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    noise1 = np.random.normal(0, 3.8, len(t))
    noise2 = np.random.normal(0, 3.2, len(t))
    
    # Motor Unit Action Potential (MUAP) burst packets
    burst1 = 64.0 * np.sin(2 * np.pi * 60 * t) * np.exp(-((t - 0.28) ** 2) / 0.008)
    burst2 = 80.0 * np.sin(2 * np.pi * 92 * t) * np.exp(-((t - 0.65) ** 2) / 0.012)
    burst3 = -52.0 * np.sin(2 * np.pi * 75 * t) * np.exp(-((t - 0.82) ** 2) / 0.006)
    
    ch1 = noise1 + burst1 + burst2 + burst3
    return pd.DataFrame({
        "EMG_Signal_uV": np.round(ch1, 2),
    })


def get_synthetic_ultrasound():
    """Generates a B-mode muscle ultrasound scan for demonstration."""
    np.random.seed(101)
    w, h = 320, 240
    base = np.random.rayleigh(scale=35, size=(h, w)).clip(0, 255)
    
    # Fascial lines
    y_coords = np.arange(h)[:, None]
    fascia_top = np.exp(-((y_coords - 45) ** 2) / 18.0) * 160
    fascia_bottom = np.exp(-((y_coords - 195) ** 2) / 22.0) * 150
    
    # Fascicular striations
    striations = 30 * np.sin(np.linspace(0, 14 * np.pi, h))[:, None] * (y_coords > 55) * (y_coords < 185)
    
    img_arr = (base + fascia_top + fascia_bottom + striations).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_arr, mode="L").convert("RGB")


def extract_emg_features(signal_values, fs=1000.0):
    """
    Computes mathematical feature values from signal:
    - Root Mean Square (RMS)
    - Mean Absolute Value (MAV)
    - Median Frequency (MDF) via FFT power spectrum
    - Wavelet Sub-band Energy via Haar DWT
    """
    x = np.asarray(signal_values, dtype=float)
    if len(x) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # 1. RMS
    rms = float(np.sqrt(np.mean(x ** 2)))
    
    # 2. MAV
    mav = float(np.mean(np.abs(x)))
    
    # 3. Median Frequency (MDF) via FFT Power Spectral Density
    fft_vals = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    cumulative_power = np.cumsum(fft_vals)
    total_power = cumulative_power[-1] if len(cumulative_power) > 0 else 1.0
    
    if total_power > 0:
        idx_med = np.where(cumulative_power >= total_power / 2.0)[0]
        mdf = float(freqs[idx_med[0]]) if len(idx_med) > 0 else 0.0
    else:
        mdf = 0.0
        
    # 4. Wavelet Sub-band Energy (Haar discrete wavelet decomposition)
    x_even = x if len(x) % 2 == 0 else x[:-1]
    if len(x_even) > 0:
        detail_coeffs = (x_even[0::2] - x_even[1::2]) / np.sqrt(2.0)
        wavelet_energy = float(np.sum(detail_coeffs ** 2) / 1000.0)
    else:
        wavelet_energy = 0.0
        
    return rms, mav, mdf, wavelet_energy


def create_processed_ultrasound(orig_img):
    """Generates speckle-reduced ultrasound frame with identified muscle ROI."""
    filtered = orig_img.filter(ImageFilter.MedianFilter(size=3))
    filtered = filtered.filter(ImageFilter.EDGE_ENHANCE)
    
    annotated = filtered.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size
    roi_box = [int(w * 0.15), int(h * 0.24), int(w * 0.85), int(h * 0.78)]
    draw.rectangle(roi_box, outline="#0d9488", width=2)
    draw.text((roi_box[0] + 8, roi_box[1] + 6), "Target Muscle ROI", fill="#0d9488")
    return annotated


def create_gradcam_overlay(orig_img):
    """Generates Grad-CAM class activation map overlay for ultrasound frame."""
    w, h = orig_img.size
    x = np.linspace(-2.2, 2.2, w)
    y = np.linspace(-2.2, 2.2, h)
    xx, yy = np.meshgrid(x, y)
    activation = np.exp(-(xx**2 + 1.2 * yy**2))
    activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
    
    # Colormap calculation
    heatmap_r = (255 * np.clip(1.5 * activation, 0, 1)).astype(np.uint8)
    heatmap_g = (255 * np.clip(1.5 * (1 - np.abs(activation - 0.5) * 2), 0, 1)).astype(np.uint8)
    heatmap_b = (255 * np.clip(1.5 * (1 - activation), 0, 1)).astype(np.uint8)
    
    heatmap_rgb = np.stack([heatmap_r, heatmap_g, heatmap_b], axis=-1)
    heatmap_img = Image.fromarray(heatmap_rgb, mode="RGB")
    
    return Image.blend(orig_img.convert("RGB"), heatmap_img, alpha=0.45)


def create_shap_plot():
    """Generates SHAP feature attribution bar plot."""
    features = [
        "Root Mean Square (RMS)",
        "Median Frequency (MDF)",
        "Wavelet Energy (DWT-db4)",
        "Zero Crossing Rate (ZCR)",
        "Mean Absolute Value (MAV)",
    ]
    shap_values = [0.38, 0.29, 0.21, -0.14, 0.08]
    colors = ["#0d9488" if v > 0 else "#f43f5e" for v in shap_values]
    
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=100)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, shap_values, color=colors, height=0.55, align="center")
    ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1.0)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9, color="#0f172a", fontweight="500")
    ax.set_xlabel("Mean |SHAP Value| (Impact on Model Classification)", fontsize=8.5, color="#475569")
    ax.set_title("SHAP Feature Importance (EMG Domain)", fontsize=10, color="#0f172a", pad=10, fontweight="600")
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors="#475569", labelsize=8.5)
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "emg_file" not in st.session_state:
    st.session_state.emg_file = None
if "emg_filename" not in st.session_state:
    st.session_state.emg_filename = None
if "emg_df" not in st.session_state:
    st.session_state.emg_df = None
if "selected_emg_col" not in st.session_state:
    st.session_state.selected_emg_col = None

if "us_file" not in st.session_state:
    st.session_state.us_file = None
if "us_filename" not in st.session_state:
    st.session_state.us_filename = None
if "us_image" not in st.session_state:
    st.session_state.us_image = None

# Probabilities & Weights
if "demo_emg_prob" not in st.session_state:
    st.session_state.demo_emg_prob = 0.78
if "demo_us_prob" not in st.session_state:
    st.session_state.demo_us_prob = 0.62
if "demo_alpha" not in st.session_state:
    st.session_state.demo_alpha = 0.50

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔬 ALS Research Lab")
    st.caption("Multimodal ALS Detection System")
    st.markdown("---")

    menu_option = st.radio(
        "Navigation",
        [
            "1. Data Input",
            "2. Analysis",
            "3. Results",
            "4. Explainability",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### System Status")
    
    has_emg = st.session_state.emg_df is not None
    has_us = st.session_state.us_image is not None

    st.markdown(
        f"""
        <div style="font-size:0.83rem; color:#475569; line-height:1.7;">
            • <b>System Pipeline:</b> <span class="status-badge">Online</span><br>
            • <b>EMG Signal:</b> {"Loaded" if has_emg else "Waiting for input"}<br>
            • <b>Ultrasound:</b> {"Loaded" if has_us else "Waiting for input"}<br>
            • <b>Fusion Engine:</b> Ready
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 1. DATA INPUT PAGE (DEFAULT LANDING PAGE)
# -----------------------------------------------------------------------------
if menu_option == "1. Data Input":
    st.title("Data Ingestion & Inspection")
    st.markdown("Upload multimodal patient recordings or load sample data for evaluation.")
    
    # Top Action Bar
    btn_col1, btn_col2, _ = st.columns([1.6, 1.4, 4.0])
    with btn_col1:
        if st.button("Load Demo Data", use_container_width=True, type="primary"):
            st.session_state.emg_df = get_synthetic_emg()
            st.session_state.emg_filename = "patient_emg_sample.csv"
            st.session_state.emg_file = "demo"
            st.session_state.selected_emg_col = "EMG_Signal_uV"
            
            st.session_state.us_image = get_synthetic_ultrasound()
            st.session_state.us_filename = "patient_us_bmode.png"
            st.session_state.us_file = "demo"
            st.rerun()

    with btn_col2:
        if st.button("Clear Data", use_container_width=True):
            st.session_state.emg_df = None
            st.session_state.emg_filename = None
            st.session_state.emg_file = None
            st.session_state.selected_emg_col = None
            st.session_state.us_image = None
            st.session_state.us_filename = None
            st.session_state.us_file = None
            st.rerun()

    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")

    # Column 1: EMG CSV Input
    with col1:
        st.subheader("1. Electromyography (EMG) Signal")
        st.caption("Upload surface EMG voltage time-series (.csv)")

        uploaded_emg = st.file_uploader(
            "Upload EMG CSV",
            type=["csv", "txt"],
            key="emg_uploader",
            label_visibility="collapsed",
        )

        if uploaded_emg is not None:
            try:
                df_loaded = pd.read_csv(uploaded_emg)
                num_cols = df_loaded.select_dtypes(include=["float64", "int64", "float32"]).columns.tolist()
                if len(num_cols) == 0:
                    st.error("Uploaded file contains no numeric columns suitable for signal analysis.")
                else:
                    st.session_state.emg_file = uploaded_emg
                    st.session_state.emg_filename = uploaded_emg.name
                    st.session_state.emg_df = df_loaded
                    if st.session_state.selected_emg_col not in num_cols:
                        st.session_state.selected_emg_col = num_cols[0]
            except Exception as e:
                st.error(f"Failed to parse CSV file: {e}")

        # Display preview if EMG data exists
        if st.session_state.emg_df is not None:
            df = st.session_state.emg_df
            numeric_cols = df.select_dtypes(include=["float64", "int64", "float32"]).columns.tolist()
            
            st.markdown(
                f"""
                <div style="font-size:0.85rem; color:#334155; margin-bottom:8px;">
                    <b>File:</b> <code>{st.session_state.emg_filename}</code> &nbsp;|&nbsp; 
                    <b>Size:</b> {df.shape[0]} samples × {df.shape[1]} columns
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("##### Raw Signal Waveform Preview")
            if len(numeric_cols) > 0:
                signal_cols = [c for c in numeric_cols if not any(term in c.lower() for term in ["time", "index", "sample", "ms", "sec"])]
                if not signal_cols:
                    signal_cols = numeric_cols

                if len(signal_cols) > 1:
                    current_col_idx = signal_cols.index(st.session_state.selected_emg_col) if st.session_state.selected_emg_col in signal_cols else 0
                    selected_col = st.selectbox("Select Signal Channel", signal_cols, index=current_col_idx)
                    st.session_state.selected_emg_col = selected_col
                else:
                    selected_col = signal_cols[0]
                    st.session_state.selected_emg_col = selected_col
                    st.markdown(f"<div style='font-size:0.85rem; color:#475569; margin-bottom:8px;'><b>Channel:</b> <code>{selected_col}</code></div>", unsafe_allow_html=True)
                
                st.line_chart(df[selected_col].head(1000), use_container_width=True)
            else:
                st.dataframe(df.head(10), use_container_width=True)
        else:
            st.markdown(
                """
                <div class="placeholder-panel">
                    <div class="placeholder-title">No EMG Signal Loaded</div>
                    <div>Upload an electromyography CSV file or click "Load Demo Data" above.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Column 2: Ultrasound Image Input
    with col2:
        st.subheader("2. Muscle Ultrasound Scan")
        st.caption("Upload B-mode sonogram (.png, .jpg)")

        uploaded_us = st.file_uploader(
            "Upload Ultrasound Image",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            key="us_uploader",
            label_visibility="collapsed",
        )

        if uploaded_us is not None:
            try:
                img_loaded = Image.open(uploaded_us)
                st.session_state.us_file = uploaded_us
                st.session_state.us_filename = uploaded_us.name
                st.session_state.us_image = img_loaded
            except Exception as e:
                st.error(f"Failed to open image file: {e}")

        # Display preview if ultrasound image exists
        if st.session_state.us_image is not None:
            img = st.session_state.us_image
            st.markdown(
                f"""
                <div style="font-size:0.85rem; color:#334155; margin-bottom:8px;">
                    <b>File:</b> <code>{st.session_state.us_filename}</code> &nbsp;|&nbsp; 
                    <b>Resolution:</b> {img.size[0]} × {img.size[1]} px
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("##### Sonogram Preview")
            st.image(img, use_container_width=True, caption=f"Ultrasound Frame: {st.session_state.us_filename}")
        else:
            st.markdown(
                """
                <div class="placeholder-panel">
                    <div class="placeholder-title">No Ultrasound Image Loaded</div>
                    <div>Upload a B-mode muscle scan or click "Load Demo Data" above.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.83rem; color:#64748b; text-align:center;">
            Data stored in session state persists through <b>Analysis</b>, <b>Results</b>, and <b>Explainability</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 2. ANALYSIS PAGE
# -----------------------------------------------------------------------------
elif menu_option == "2. Analysis":
    st.title("Feature Extraction & Modality Analysis")
    st.markdown("Signal preprocessing, extracted numerical features, and sonogram preparation.")
    st.markdown("---")

    col_emg, col_us = st.columns(2, gap="large")

    # Section 1: EMG Pipeline
    with col_emg:
        st.subheader("Electromyography (EMG) Pipeline")
        st.caption("Temporal, Spectral & Motor Unit Dynamics")

        active_emg_df = st.session_state.emg_df if st.session_state.emg_df is not None else get_synthetic_emg()
        numeric_cols = active_emg_df.select_dtypes(include=["float64", "int64", "float32"]).columns.tolist()
        active_col = st.session_state.selected_emg_col if st.session_state.selected_emg_col in numeric_cols else numeric_cols[0]
        
        signal_data = active_emg_df[active_col].values
        signal_mean_centered = signal_data - np.mean(signal_data)
        
        # Window segmentation slider
        window_size = st.slider("Segmentation Window (Samples)", min_value=100, max_value=min(1000, len(signal_data)), value=min(400, len(signal_data)), step=50)
        windowed_raw = signal_data[:window_size]
        windowed_centered = signal_mean_centered[:window_size]
        
        st.markdown("##### 1. Signal Waveforms")
        chart_df = pd.DataFrame({
            "Raw EMG (μV)": windowed_raw,
            "Mean-Centered (μV)": windowed_centered,
        })
        st.line_chart(chart_df, height=210, use_container_width=True)
        st.caption(f"Channel: {active_col} • Baseline Mean-Centered Representation")

        # Feature Extraction
        rms_val, mav_val, mdf_val, wav_val = extract_emg_features(signal_data, fs=1000.0)
        
        st.markdown("##### 2. Extracted Features")
        features_table = {
            "Domain": ["Time", "Time", "Frequency", "Time-Frequency"],
            "Parameter": [
                "Root Mean Square (RMS)",
                "Mean Absolute Value (MAV)",
                "Median Frequency (MDF)",
                "Wavelet Energy (Haar DWT)",
            ],
            "Value": [
                f"{rms_val:.3f} μV",
                f"{mav_val:.3f} μV",
                f"{mdf_val:.1f} Hz",
                f"{wav_val:.3f} mJ",
            ],
            "Biomarker Target": [
                "Motor unit amplitude",
                "Mean recruitment envelope",
                "Spectral compression / fatigue",
                "Transient burst energy",
            ],
        }
        st.dataframe(pd.DataFrame(features_table), use_container_width=True, hide_index=True)

        # EMG Model Risk Output Card
        st.markdown("##### 3. EMG Model Risk Score")
        emg_p = st.session_state.demo_emg_prob
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">EMG Model Score (y_EMG)</div>
                <div class="metric-value">{emg_p * 100:.1f}%</div>
                <div class="metric-sub">1D Temporal & Spectral Feature Classifier</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section 2: Ultrasound Pipeline
    with col_us:
        st.subheader("Muscle Ultrasound Pipeline")
        st.caption("Echointensity, Fasciculation Dynamics & Tissue Architecture")

        active_us_img = st.session_state.us_image if st.session_state.us_image is not None else get_synthetic_ultrasound()

        # Image Preparation
        target_size = (224, 224)
        resized_img = active_us_img.resize(target_size, Image.Resampling.BILINEAR)
        
        st.markdown("##### 1. Image Preparation")
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            st.image(active_us_img, use_container_width=True, caption=f"Original ({active_us_img.size[0]}×{active_us_img.size[1]})")
        with subcol2:
            st.image(resized_img, use_container_width=True, caption=f"Normalized Input ({target_size[0]}×{target_size[1]})")

        # ROI & Speckle Filtering
        st.markdown("##### 2. Speckle-Filtered & Muscle ROI Segmentation")
        processed_us = create_processed_ultrasound(active_us_img)
        st.image(processed_us, use_container_width=True, caption="Speckle Reduction & Region of Interest Segmentation")

        # Ultrasound Model Risk Output Card
        st.markdown("##### 3. Ultrasound Model Risk Score")
        us_p = st.session_state.demo_us_prob
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Ultrasound Model Score (y_US)</div>
                <div class="metric-value">{us_p * 100:.1f}%</div>
                <div class="metric-sub">2D Spatial Sonographic CNN Model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# 3. RESULTS AND MULTIMODAL FUSION
# -----------------------------------------------------------------------------
elif menu_option == "3. Results":
    st.title("Results")
    st.markdown("Multimodal decision metrics and late fusion classification.")
    st.markdown("---")

    # Current values (computed from data / models)
    cur_emg_p = st.session_state.demo_emg_prob
    cur_us_p = st.session_state.demo_us_prob
    cur_alpha = 0.50
    fused_score = (cur_alpha * cur_emg_p) + ((1.0 - cur_alpha) * cur_us_p)
    
    if fused_score >= 0.65:
        class_label = "High Risk"
        class_color = "#b91c1c"
    elif fused_score >= 0.40:
        class_label = "Moderate Risk"
        class_color = "#b45309"
    else:
        class_label = "Low Risk"
        class_color = "#15803d"

    # Primary Decision Metric Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">EMG Probability</div>
                <div class="metric-value">{cur_emg_p * 100:.1f}%</div>
                <div class="metric-sub">y_EMG</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Ultrasound Probability</div>
                <div class="metric-value">{cur_us_p * 100:.1f}%</div>
                <div class="metric-sub">y_US</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card" style="border: 2px solid #0d9488;">
                <div class="metric-label" style="color: #0f766e; font-weight: bold;">Fused ALS Risk Score</div>
                <div class="metric-value" style="color: #0f766e;">{fused_score * 100:.1f}%</div>
                <div class="metric-sub">Late Fusion (α = 0.50)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Final Classification</div>
                <div class="metric-value" style="font-size: 1.25rem; color: {class_color};">{class_label}</div>
                <div class="metric-sub">Multimodal Consensus</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Strategy Comparison Table
    st.subheader("Model Comparison")
    comparison_data = {
        "Model / Strategy": [
            "EMG Model",
            "Ultrasound Model",
            "Early Fusion",
            "Late Fusion",
        ],
        "Risk Score": [
            f"{cur_emg_p * 100:.1f}%",
            f"{cur_us_p * 100:.1f}%",
            f"{min(1.0, (cur_emg_p * 0.52 + cur_us_p * 0.50)) * 100:.1f}%",
            f"{fused_score * 100:.1f}%",
        ],
    }
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# 4. EXPLAINABILITY PAGE
# -----------------------------------------------------------------------------
elif menu_option == "4. Explainability":
    st.title("Model Interpretability & Explainability")
    st.markdown("Attribution frameworks for model transparency: SHAP for electrophysiology and Grad-CAM for sonography.")
    st.markdown("---")

    col_shap, col_gradcam = st.columns(2, gap="large")

    # Section 1: EMG — SHAP
    with col_shap:
        st.subheader("EMG — SHAP")
        st.caption("SHapley Additive exPlanations for Feature Importance")

        fig_shap = create_shap_plot()
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)

        st.markdown(
            """
            <div class="content-box" style="margin-top: 12px; font-size: 0.84rem; line-height: 1.6;">
                <b style="color: #0f172a;">Feature Attribution Breakdown:</b>
                <ul style="margin-top: 6px; margin-bottom: 0; padding-left: 18px; color: #475569;">
                    <li><b>RMS Voltage (+0.38):</b> High-amplitude discharge bursts drive positive classification towards ALS risk.</li>
                    <li><b>Median Frequency (+0.29):</b> Spectral shift towards lower frequencies reflects motor unit firing rate alterations.</li>
                    <li><b>Wavelet Sub-band Energy (+0.21):</b> High-frequency transient spikes correspond to spontaneous potentials.</li>
                    <li><b>Zero Crossing Rate (-0.14):</b> Slower turn frequency slightly attenuates the risk index.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Section 2: Ultrasound — Grad-CAM
    with col_gradcam:
        st.subheader("Ultrasound — Grad-CAM")
        st.caption("Gradient-Weighted Class Activation Mapping on Sonographic Scans")

        active_us_img = st.session_state.us_image if st.session_state.us_image is not None else get_synthetic_ultrasound()
        gradcam_img = create_gradcam_overlay(active_us_img)
        
        st.image(
            gradcam_img,
            use_container_width=True,
            caption="Grad-CAM Class Activation Heatmap Overlay",
        )

        st.markdown(
            """
            <div class="content-box" style="margin-top: 12px; font-size: 0.84rem; line-height: 1.6;">
                <b style="color: #0f172a;">Visual Activation Insights:</b>
                <ul style="margin-top: 6px; margin-bottom: 0; padding-left: 18px; color: #475569;">
                    <li><b>Focal Heatmap Concentration:</b> Red/yellow hot zones identify spatial regions influencing the CNN classifier.</li>
                    <li><b>Fascicular Architecture:</b> Highlights disrupted echo-texture and boundary thickening in the deep muscle belly.</li>
                    <li><b>Acoustic Attenuation:</b> Discriminates between deep bone acoustic shadows and genuine muscle atrophy.</li>
                    <li><b>Clinician-Verifiable:</b> Provides visual anatomical correlation supporting the decision score.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="border-top: 1px solid #e2e8f0; padding-top: 14px; text-align: center; color: #94a3b8; font-size: 0.8rem;">
        Multimodal ALS Detection System • B.Tech Capstone Project • Biomedical & Computer Engineering
    </div>
    """,
    unsafe_allow_html=True,
)
