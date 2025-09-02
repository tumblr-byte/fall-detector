import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import io

# Configure page
st.set_page_config(
    page_title="Fall Detection System",
    page_icon="🚨",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #ff4b4b;
        margin-bottom: 30px;
    }
    .demo-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 15px;
        margin: 20px 0;
    }
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 18px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Class definitions and colors
CLASSES = ["Fall Detected", "Walking", "Sitting"]
CLASS_COLORS = {
    "Fall Detected": (255, 0, 0),
    "Walking": (0, 255, 0),
    "Sitting": (0, 0, 255)
}

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")  # Update this path
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def draw_predictions(image, results):
    img_copy = image.copy()
    
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Class {cls_id}"
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(img_copy, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(img_copy, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    return img_copy

def process_image(uploaded_file, model):
    try:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        results = model.predict(source=img_array, conf=0.6, verbose=False)
        result_img = draw_predictions(img_array, results)
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        return result_img_rgb, results
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def process_video(uploaded_file, model):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        cap = cv2.VideoCapture(temp_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, conf=0.6, verbose=False)
            result_frame = draw_predictions(frame, results)
            out.write(result_frame)
            
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
        
        cap.release()
        out.release()
        os.unlink(temp_path)
        return output_path
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

def main():
    st.markdown("<h1 class='main-header'>🚨 Fall Detection System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='demo-warning'>
        <h3>Demo Testing Version</h3>
        <p>This is a small demo for testing purposes</p>
        <p><strong>Detection Classes:</strong> Fall Detected, Walking, Sitting</p>
    </div>
    """, unsafe_allow_html=True)
    
    model = load_model()
    if model is None:
        st.error("Failed to load YOLO model. Please check the model path.")
        return
    
    st.sidebar.title("Detection Options")
    detection_mode = st.sidebar.selectbox(
        "Choose Detection Mode:",
        ["📷 Image Upload", "🎥 Video Upload"]
    )
    
    # ---------------- Image Detection ----------------
    if detection_mode == "📷 Image Upload":
        st.header("Image Detection")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to detect falls, walking, or sitting"
        )
        if uploaded_file is not None:
            file_size = len(uploaded_file.read())
            uploaded_file.seek(0)
            if file_size > 15 * 1024 * 1024:
                st.error("❌ File too large! Please upload an image smaller than 15 MB.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(Image.open(uploaded_file), use_column_width=True)
                with col2:
                    st.subheader("Detection Results")
                    with st.spinner("Processing image..."):
                        result_img, results = process_image(uploaded_file, model)
                    if result_img is not None:
                        st.image(result_img, use_column_width=True)
                        buf = io.BytesIO()
                        Image.fromarray(result_img).save(buf, format='PNG')
                        st.download_button(
                            label="📥 Download Result Image",
                            data=buf.getvalue(),
                            file_name="fall_detection_result.png",
                            mime="image/png"
                        )
                        if results and len(results[0].boxes) > 0:
                            st.success(f"Detected {len(results[0].boxes)} objects")
                            for i, box in enumerate(results[0].boxes):
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Class {cls_id}"
                                st.write(f"• {class_name}: {conf:.2f} confidence")
                        else:
                            st.info("No objects detected")
    
    # ---------------- Video Detection ----------------
    elif detection_mode == "🎥 Video Upload":
        st.header("Video Detection")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video to detect falls, walking, or sitting"
        )
        if uploaded_file is not None:
            file_size = len(uploaded_file.read())
            uploaded_file.seek(0)
            if file_size > 15 * 1024 * 1024:
                st.error("❌ File too large! Please upload a video smaller than 15 MB.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Video")
                    st.video(uploaded_file)
                with col2:
                    st.subheader("Process Video")
                    if st.button("🚀 Start Detection", key="process_video"):
                        with st.spinner("Processing video... This may take a while."):
                            output_path = process_video(uploaded_file, model)
                        if output_path and os.path.exists(output_path):
                            st.success("Video processing completed!")
                            with open(output_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                                st.video(video_bytes)
                                st.download_button(
                                    label="📥 Download Result Video",
                                    data=video_bytes,
                                    file_name="fall_detection_result.mp4",
                                    mime="video/mp4"
                                )
                            os.unlink(output_path)
    
    # ---------------- Sidebar & Footer ----------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Detection Classes")
    for class_name, color in CLASS_COLORS.items():
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        st.sidebar.markdown(f"🔸 **{class_name}** - <span style='color:{color_hex}'></span>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Instructions")
    st.sidebar.markdown("""
    1. **Image**: Upload JPG/PNG files
    2. **Video**: Upload MP4/AVI/MOV files  
    3. **Download**: Get processed results
    """)
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Fall Detection Demo | Built with YOLO & Streamlit</p>
        <p>This is a testing demo - Results may vary</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
