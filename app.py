import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import io
import time

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
    "Fall Detected": (255, 0, 0),    # Red
    "Walking": (0, 255, 0),          # Green  
    "Sitting": (0, 0, 255)           # Blue
}

# Load YOLO model
@st.cache_resource
def load_model():
    try:
        # You'll need to update this path to your model
        model = YOLO("best.pt")  # Update this path
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def draw_predictions(image, results):
    """Draw bounding boxes and labels on image"""
    img_copy = image.copy()
    
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Get class name and color
            class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Class {cls_id}"
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{class_name}: {conf:.2f}"
            
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(img_copy, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(img_copy, label, (x1, y1 - 5), 
                       font, font_scale, (255, 255, 255), thickness)
    
    return img_copy

def process_image(uploaded_file, model):
    """Process uploaded image"""
    try:
        # Read image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Run prediction
        results = model.predict(source=img_array, conf=0.6, verbose=False)
        
        # Draw predictions
        result_img = draw_predictions(img_array, results)
        
        # Convert back to RGB for display
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        return result_img_rgb, results
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def process_video(uploaded_file, model):
    """Process uploaded video"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # Open video
        cap = cv2.VideoCapture(temp_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Prepare output video
        output_path = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run prediction on frame
            results = model.predict(source=frame, conf=0.6, verbose=False)
            
            # Draw predictions
            result_frame = draw_predictions(frame, results)
            
            # Write frame to output video
            out.write(result_frame)
            
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
        
        cap.release()
        out.release()
        
        # Clean up temp input file
        os.unlink(temp_path)
        
        return output_path
        
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

def main():
    st.markdown("<h1 class='main-header'>🚨 Fall Detection System</h1>", unsafe_allow_html=True)
    
    # Demo warning
    st.markdown("""
    <div class='demo-warning'>
        <h3>⚠️ Demo Testing Version</h3>
        <p>This is a small demo for testing purposes. Please ensure uploaded files are less than <strong>15MB</strong> for optimal performance.</p>
        <p><strong>Detection Classes:</strong> Fall Detected, Walking, Sitting</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load YOLO model. Please check the model path.")
        return
    
    # Sidebar for options
    st.sidebar.title("Detection Options")
    detection_mode = st.sidebar.selectbox(
        "Choose Detection Mode:",
        ["📷 Image Upload", "🎥 Video Upload", "📹 Live Webcam"]
    )
    
    if detection_mode == "📷 Image Upload":
        st.header("Image Detection")
        
        uploaded_file = st.file_uploader(
            "Choose an image file (Max 15MB)",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to detect falls, walking, or sitting"
        )
        
        if uploaded_file is not None:
            # Check file size
            file_size = len(uploaded_file.read())
            uploaded_file.seek(0)  # Reset file pointer
            
            if file_size > 15 * 1024 * 1024:  # 15MB
                st.error("File size exceeds 15MB limit. Please upload a smaller image.")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                original_image = Image.open(uploaded_file)
                st.image(original_image, use_column_width=True)
            
            with col2:
                st.subheader("Detection Results")
                with st.spinner("Processing image..."):
                    result_img, results = process_image(uploaded_file, model)
                
                if result_img is not None:
                    st.image(result_img, use_column_width=True)
                    
                    # Download button
                    result_pil = Image.fromarray(result_img)
                    buf = io.BytesIO()
                    result_pil.save(buf, format='PNG')
                    
                    st.download_button(
                        label="📥 Download Result Image",
                        data=buf.getvalue(),
                        file_name="fall_detection_result.png",
                        mime="image/png"
                    )
                    
                    # Show detection summary
                    if results and len(results[0].boxes) > 0:
                        st.success(f"Detected {len(results[0].boxes)} objects")
                        for i, box in enumerate(results[0].boxes):
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Class {cls_id}"
                            st.write(f"• {class_name}: {conf:.2f} confidence")
                    else:
                        st.info("No objects detected")
    
    elif detection_mode == "🎥 Video Upload":
        st.header("Video Detection")
        
        uploaded_file = st.file_uploader(
            "Choose a video file (Max 15MB)",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video to detect falls, walking, or sitting"
        )
        
        if uploaded_file is not None:
            # Check file size
            file_size = len(uploaded_file.read())
            uploaded_file.seek(0)  # Reset file pointer
            
            if file_size > 15 * 1024 * 1024:  # 15MB
                st.error("File size exceeds 15MB limit. Please upload a smaller video.")
                return
            
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
                        
                        # Display result video
                        with open(output_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes)
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Result Video",
                                data=video_bytes,
                                file_name="fall_detection_result.mp4",
                                mime="video/mp4"
                            )
                        
                        # Clean up
                        os.unlink(output_path)
    
    elif detection_mode == "📹 Live Webcam":
        st.header("Live Webcam Detection")
        
        # Instructions
        st.info("Click 'Start Live Detection' to begin real-time fall detection using your webcam.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            start_detection = st.button("🎥 Start Live Detection", key="start_live")
        
        with col2:
            stop_detection = st.button("⏹️ Stop Detection", key="stop_live")
        
        # Placeholder for webcam feed
        frame_placeholder = st.empty()
        
        if start_detection:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not access webcam. Please check your camera permissions.")
                return
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            st.success("Live detection started! Press 'Stop Detection' to end.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam")
                    break
                
                # Run prediction
                results = model.predict(source=frame, conf=0.6, verbose=False)
                
                # Draw predictions
                result_frame = draw_predictions(frame, results)
                
                # Convert BGR to RGB for display
                result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(result_frame_rgb, channels="RGB", use_column_width=True)
                
                # Check if stop button was pressed
                if stop_detection:
                    break
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
            
            cap.release()
            st.info("Live detection stopped.")
    
    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Detection Classes")
    for class_name, color in CLASS_COLORS.items():
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        st.sidebar.markdown(f"🔸 **{class_name}** - <span style='color:{color_hex}'>●</span>", unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Instructions")
    st.sidebar.markdown("""
    1. **Image**: Upload JPG/PNG files
    2. **Video**: Upload MP4/AVI/MOV files  
    3. **Live**: Use your webcam for real-time detection
    4. **File Limit**: Maximum 15MB per file
    5. **Download**: Get processed results
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 Fall Detection Demo | Built with YOLO & Streamlit</p>
        <p>⚠️ This is a testing demo - Results may vary</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()