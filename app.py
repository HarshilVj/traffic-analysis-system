import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import os
from pathlib import Path
import tempfile
os.environ["YOLO_CPUINFO"] = "False"

# Import custom modules
from anpr_module import ANPRDetector
from vehicle_classifier import VehicleClassifier
from utils import draw_results, create_results_dataframe
from video_module import VideoProcessor

# Page configuration
st.set_page_config(
    page_title="Traffic Analysis System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .results-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'video_done' not in st.session_state:
    st.session_state.video_done = False
if 'video_results' not in st.session_state:
    st.session_state.video_results = None

def initialize_models():
    """Initialize YOLO models for ANPR and vehicle classification"""
    try:
        with st.spinner("Loading AI models... This may take a moment..."):
            anpr_detector = ANPRDetector("models/best2.pt")
            vehicle_classifier = VehicleClassifier("models/best.pt")
        st.success("✅ Models loaded successfully!")
        return anpr_detector, vehicle_classifier
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("Please ensure model files are in the 'models' directory")
        return None, None

def process_image(image, anpr_detector, vehicle_classifier):
    """Process image through both ANPR and vehicle classification"""

    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 2:  # Grayscale
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:  # RGBA
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:  # RGB
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    results = []

    # Step 1: Detect vehicles
    with st.spinner("🚗 Detecting vehicles..."):
        vehicle_detections = vehicle_classifier.detect_vehicles(img_cv)

    # Step 2: Process each vehicle
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, vehicle in enumerate(vehicle_detections):
        status_text.text(f"Processing vehicle {idx + 1}/{len(vehicle_detections)}...")
        progress_bar.progress((idx + 1) / len(vehicle_detections))

        # Get vehicle crop
        x1, y1, x2, y2 = vehicle['bbox']
        vehicle_crop = img_cv[y1:y2, x1:x2]

        # Detect number plate in vehicle region
        plate_detections = anpr_detector.detect_plates(vehicle_crop)

        for plate in plate_detections:
            # Get plate coordinates relative to original image
            px1, py1, px2, py2 = plate['bbox']
            plate_x1 = x1 + px1
            plate_y1 = y1 + py1
            plate_x2 = x1 + px2
            plate_y2 = y1 + py2

            # Extract plate crop from original image
            plate_crop = img_cv[plate_y1:plate_y2, plate_x1:plate_x2]

            # Perform OCR
            raw_text, final_plate = anpr_detector.perform_ocr(plate_crop)

            # Store results
            results.append({
                'vehicle_bbox': vehicle['bbox'],
                'vehicle_type': vehicle['vehicle_type'],
                'vehicle_class': vehicle['class_name'],
                'vehicle_confidence': vehicle['confidence'],
                'plate_bbox': (plate_x1, plate_y1, plate_x2, plate_y2),
                'plate_crop': plate_crop,
                'raw_ocr': raw_text,
                'final_plate': final_plate,
                'plate_confidence': plate['confidence']
            })

    progress_bar.empty()
    status_text.empty()

    return results, img_cv

def render_image_tab(vehicle_conf, plate_conf):
    """Single-image analysis UI (original behaviour, unchanged)."""
    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a traffic image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of vehicles with visible number plates"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Process button
            if st.button("🔍 Analyze Traffic", type="primary"):
                # Initialize models
                anpr_detector, vehicle_classifier = initialize_models()

                if anpr_detector and vehicle_classifier:
                    # Update confidence thresholds
                    vehicle_classifier.conf_threshold = vehicle_conf
                    anpr_detector.conf_threshold = plate_conf

                    # Process image
                    with st.spinner("Processing image... Please wait..."):
                        results, processed_img = process_image(
                            image, anpr_detector, vehicle_classifier
                        )

                    # Store in session state
                    st.session_state.results = results
                    st.session_state.processed_img = processed_img
                    st.session_state.processed = True

                    st.success(f"✅ Analysis complete! Found {len(results)} vehicle(s)")

    with col2:
        st.subheader("📊 Analysis Results")

        if st.session_state.processed and st.session_state.results:
            results = st.session_state.results
            processed_img = st.session_state.processed_img

            # Draw results on image
            annotated_img = draw_results(processed_img.copy(), results)

            # Convert BGR to RGB for display
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_img_rgb, caption="Analyzed Image", use_column_width=True)

            # Download button for annotated image
            _, buffer = cv2.imencode('.jpg', annotated_img)
            st.download_button(
                label="📥 Download Annotated Image",
                data=buffer.tobytes(),
                file_name="traffic_analysis_result.jpg",
                mime="image/jpeg",
                # use_column_width=True
            )
        else:
            st.info("👆 Upload an image and click 'Analyze Traffic' to see results")

    # Detailed results section
    if st.session_state.processed and st.session_state.results:
        st.markdown("---")
        st.header("📋 Detailed Results")

        results = st.session_state.results

        if len(results) == 0:
            st.warning("⚠️ No vehicles with readable number plates detected")
        else:
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["🖼️ Gallery View", "📊 Table View", "📄 Export Data"])

            with tab1:
                # Display each vehicle result
                for idx, result in enumerate(results, 1):
                    with st.expander(f"Vehicle #{idx} - {result['vehicle_type']} - {result['final_plate']}", expanded=True):
                        col_a, col_b, col_c = st.columns([1, 1, 1])

                        with col_a:
                            st.markdown("**📸 Number Plate**")
                            if result['plate_crop'] is not None and result['plate_crop'].size > 0:
                                plate_rgb = cv2.cvtColor(result['plate_crop'], cv2.COLOR_BGR2RGB)
                                st.image(plate_rgb, use_column_width=True)
                            else:
                                st.warning("No plate crop available")

                        with col_b:
                            st.markdown("**🚗 Vehicle Info**")
                            st.write(f"**Type:** {result['vehicle_type']}")
                            st.write(f"**Class:** {result['vehicle_class']}")
                            st.write(f"**Confidence:** {result['vehicle_confidence']:.2%}")

                        with col_c:
                            st.markdown("**🔤 OCR Results**")

                            # Status badge
                            if result['final_plate'] not in ["INVALID", "UNREADABLE"]:
                                st.success(f"✅ **{result['final_plate']}**")
                            elif result['final_plate'] == "INVALID":
                                st.warning(f"⚠️ **INVALID**")
                            else:
                                st.error(f"❌ **UNREADABLE**")

                            st.write(f"**Raw OCR:** {result['raw_ocr']}")
                            st.write(f"**Plate Conf:** {result['plate_confidence']:.2%}")

            with tab2:
                # Create DataFrame
                df = create_results_dataframe(results)
                st.dataframe(df.reset_index(drop=True), use_container_width=True)

                # Statistics
                st.markdown("### 📈 Statistics")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)

                with col_s1:
                    st.metric("Total Vehicles", len(results))
                with col_s2:
                    valid_plates = sum(1 for r in results if r['final_plate'] not in ["INVALID", "UNREADABLE"])
                    st.metric("Valid Plates", valid_plates)
                with col_s3:
                    vehicle_types = pd.Series([r['vehicle_type'] for r in results])
                    st.metric("Most Common", vehicle_types.mode()[0] if len(vehicle_types) > 0 else "N/A")
                with col_s4:
                    avg_conf = np.mean([r['vehicle_confidence'] for r in results])
                    st.metric("Avg Confidence", f"{avg_conf:.1%}")

            with tab3:
                st.markdown("### 💾 Export Options")

                # CSV Export
                df = create_results_dataframe(results)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="traffic_analysis_results.csv",
                    mime="text/csv",
                    # use_column_width=True
                )

                # JSON Export
                import json
                json_data = []
                for r in results:
                    json_data.append({
                        'vehicle_type': r['vehicle_type'],
                        'vehicle_class': r['vehicle_class'],
                        'vehicle_confidence': float(r['vehicle_confidence']),
                        'plate_number': r['final_plate'],
                        'raw_ocr': r['raw_ocr'],
                        'plate_confidence': float(r['plate_confidence'])
                    })

                json_str = json.dumps(json_data, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name="traffic_analysis_results.json",
                    mime="application/json",
                    # use_column_width=True
                )

def render_video_tab(vehicle_conf, plate_conf):
    """Browser plays the original video smoothly; detection runs decoupled."""
    st.subheader("📹 Upload Road Video")
    st.caption(
        "The video plays smoothly in your browser (original form, no overlays). "
        "Detection runs separately and clear-plate captures stream in below as "
        "they're found."
    )

    uploaded_video = st.file_uploader(
        "Choose a traffic video...",
        type=["mp4", "avi", "mov", "mkv"],
        help="MP4 (H.264) plays back smoothest in the browser. "
             "Short clips (≤ ~30s, 720p) process fastest on Streamlit Cloud's CPU."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        scan_stride = st.slider(
            "Scan every Nth frame", 1, 10, 3,
            help="How often to look for clear plates. Lower = more thorough, slower."
        )
    with col_b:
        clarity_conf = st.slider(
            "Plate clarity (min confidence)", 0.30, 0.95, 0.60, 0.05,
            help="Only act on plates detected at least this confidently."
        )
    with col_c:
        min_sharpness = st.slider(
            "Min sharpness (anti-blur)", 0, 300, 80, 10,
            help="Higher = require sharper, less motion-blurred plates."
        )

    if uploaded_video is not None and st.button("▶️ Play & Capture", type="primary"):
        # Read the upload once: feed the bytes to the browser player AND to OpenCV
        video_bytes = uploaded_video.getvalue()

        # Smooth playback — handled entirely by the browser, zero server cost
        st.markdown("#### ▶️ Playback (original, plays in your browser)")
        st.video(video_bytes)

        # Save to a temp file so OpenCV can read frames for processing
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_bytes)
        tfile.close()

        anpr_detector, vehicle_classifier = initialize_models()
        if not (anpr_detector and vehicle_classifier):
            return

        vehicle_classifier.conf_threshold = vehicle_conf
        anpr_detector.conf_threshold = plate_conf

        processor = VideoProcessor(
            anpr_detector, vehicle_classifier,
            clarity_conf=clarity_conf, min_sharpness=float(min_sharpness)
        )

        cap = cv2.VideoCapture(tfile.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        st.markdown("#### 📸 Latest Capture")
        capture_ph = st.empty()
        st.markdown("#### 📋 Captured So Far")
        progress = st.progress(0)
        status = st.empty()
        table_ph = st.empty()

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only the scanned frames do any heavy work; the video keeps playing
            # smoothly in the browser regardless of how long this takes.
            if idx % scan_stride == 0:
                captures = processor.scan_frame(frame)
                if captures:
                    latest = captures[-1]
                    with capture_ph.container():
                        cc1, cc2 = st.columns([1, 2])
                        with cc1:
                            if latest['plate_crop'] is not None and latest['plate_crop'].size > 0:
                                st.image(
                                    cv2.cvtColor(latest['plate_crop'], cv2.COLOR_BGR2RGB),
                                    use_column_width=True
                                )
                        with cc2:
                            st.success(
                                f"📸 Track {latest['track_id']}: **{latest['final_plate']}**"
                            )
                            st.caption(
                                f"{latest['vehicle_type']} · plate "
                                f"{latest['plate_confidence']:.0%} · ocr "
                                f"{latest['ocr_confidence']:.0%} · sharpness "
                                f"{latest['sharpness']:.0f}"
                            )
                    res = processor.get_results()
                    if res:
                        table_ph.dataframe(
                            create_results_dataframe(res).reset_index(drop=True),
                            use_container_width=True
                        )

            if total:
                progress.progress(min(1.0, (idx + 1) / total))
            status.text(
                f"Scanned up to frame {idx}/{total} · "
                f"{len(processor.tracks)} vehicle(s) seen · "
                f"{len(processor.get_results())} captured"
            )

            idx += 1

        cap.release()
        try:
            os.unlink(tfile.name)
        except OSError:
            pass

        progress.progress(1.0)
        st.session_state.video_results = processor.get_results()
        st.session_state.video_done = True
        st.success(
            f"✅ Done! {len(st.session_state.video_results)} vehicle(s) captured clearly."
        )

    # Final results
    if st.session_state.video_done and st.session_state.video_results:
        results = st.session_state.video_results
        st.markdown("---")
        st.header("📋 Unique Vehicles")

        if len(results) == 0:
            st.warning("⚠️ No vehicles detected in this video")
            return

        tab1, tab2, tab3 = st.tabs(["🖼️ Gallery View", "📊 Table View", "📄 Export Data"])

        with tab1:
            for idx, r in enumerate(results, 1):
                with st.expander(
                    f"Vehicle #{idx} (Track {r['track_id']}) - "
                    f"{r['vehicle_type']} - {r['final_plate']}",
                    expanded=False
                ):
                    col_a, col_b, col_c = st.columns([1, 1, 1])

                    with col_a:
                        st.markdown("**📸 Number Plate**")
                        if r['plate_crop'] is not None and r['plate_crop'].size > 0:
                            plate_rgb = cv2.cvtColor(r['plate_crop'], cv2.COLOR_BGR2RGB)
                            st.image(plate_rgb, use_column_width=True)
                        else:
                            st.warning("No plate captured")

                    with col_b:
                        st.markdown("**🚗 Vehicle Info**")
                        st.write(f"**Type:** {r['vehicle_type']}")
                        st.write(f"**Class:** {r['vehicle_class']}")
                        st.write(f"**Confidence:** {r['vehicle_confidence']:.2%}")
                        st.write(f"**Seen in:** {r['frames']} frame(s)")

                    with col_c:
                        st.markdown("**🔤 OCR Results**")
                        if r['final_plate'] not in ["INVALID", "UNREADABLE"]:
                            st.success(f"✅ **{r['final_plate']}**")
                        elif r['final_plate'] == "INVALID":
                            st.warning(f"⚠️ **INVALID**")
                        else:
                            st.error(f"❌ **UNREADABLE**")
                        st.write(f"**Raw OCR:** {r['raw_ocr']}")
                        st.write(f"**Plate Conf:** {r['plate_confidence']:.2%}")
                        st.write(f"**OCR Conf:** {r['ocr_confidence']:.2%}")

        with tab2:
            df = create_results_dataframe(results)
            st.dataframe(df.reset_index(drop=True), use_container_width=True)

            st.markdown("### 📈 Statistics")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Unique Vehicles", len(results))
            with col_s2:
                valid_plates = sum(
                    1 for r in results if r['final_plate'] not in ["INVALID", "UNREADABLE"]
                )
                st.metric("Valid Plates", valid_plates)
            with col_s3:
                vehicle_types = pd.Series([r['vehicle_type'] for r in results])
                st.metric("Most Common", vehicle_types.mode()[0] if len(vehicle_types) > 0 else "N/A")
            with col_s4:
                avg_conf = np.mean([r['vehicle_confidence'] for r in results])
                st.metric("Avg Confidence", f"{avg_conf:.1%}")

        with tab3:
            st.markdown("### 💾 Export Options")
            df = create_results_dataframe(results)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="video_analysis_results.csv",
                mime="text/csv",
            )

            import json
            json_data = []
            for r in results:
                json_data.append({
                    'track_id': r['track_id'],
                    'vehicle_type': r['vehicle_type'],
                    'vehicle_class': r['vehicle_class'],
                    'vehicle_confidence': float(r['vehicle_confidence']),
                    'plate_number': r['final_plate'],
                    'raw_ocr': r['raw_ocr'],
                    'plate_confidence': float(r['plate_confidence']),
                    'ocr_confidence': float(r['ocr_confidence']),
                })
            json_str = json.dumps(json_data, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name="video_analysis_results.json",
                mime="application/json",
            )

def main():
    # Header
    st.title("🚦 Intelligent Traffic Analysis System")
    st.markdown("### Automatic Number Plate Recognition (ANPR) + Vehicle Classification")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model confidence thresholds
        st.subheader("Detection Settings")
        vehicle_conf = st.slider("Vehicle Detection Confidence", 0.0, 1.0, 0.8, 0.01)
        plate_conf = st.slider("Plate Detection Confidence", 0.0, 1.0, 0.4, 0.01)

        st.markdown("---")

        # Information
        st.subheader("ℹ️ About")
        st.info("""
        This system performs:
        - Vehicle detection & classification
        - Number plate detection
        - OCR for plate reading
        - Complete traffic analysis
        """)

        st.markdown("---")
        st.subheader("📋 Supported Vehicles")
        st.markdown("""
        - 🏍️ Two Wheeler (2W)
        - 🚗 Four Wheeler (4W)
        - 🛺 Three Wheeler (3W)
        - 🚌 Heavy Motor Vehicle (HMV)
        """)

    # Image and Video modes
    image_tab, video_tab = st.tabs(["🖼️ Image", "📹 Video"])
    with image_tab:
        render_image_tab(vehicle_conf, plate_conf)
    with video_tab:
        render_video_tab(vehicle_conf, plate_conf)

# if __name__ == "__main__":
main()
