# streamlit_app.py
import streamlit as st
import cv2
import time
import numpy as np
from PIL import Image
import threading
from drowsy_detection import DrowsyDetector
import queue
import base64

# Page configuration
st.set_page_config(
    page_title="Drowsiness Detection System",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff6b6b;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-alert {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .status-normal {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .status-drowsy {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .metric-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitDrowsyDetector:
    def __init__(self):
        self.detector = None
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.stats = {
            'total_detections': 0,
            'session_start': time.time(),
            'last_drowsy_time': None
        }
    
    def initialize_detector(self, ear_thresh, wait_time):
        """Initialize the drowsiness detector"""
        try:
            self.detector = DrowsyDetector(ear_thresh=ear_thresh, wait_time=wait_time)
            return True
        except Exception as e:
            st.error(f"Error initializing detector: {str(e)}")
            return False
    
    def initialize_camera(self, camera_source):
        """Initialize camera capture"""
        try:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(camera_source)
            if not self.cap.isOpened():
                raise Exception("Cannot open camera")
            return True
        except Exception as e:
            st.error(f"Camera initialization failed: {str(e)}")
            return False
    
    def process_frame(self):
        """Process a single frame"""
        if self.cap is None or not self.cap.isOpened():
            return None, None, None
        
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None
        
        # Process frame for drowsiness detection
        ear = self.detector.process(frame)
        
        # Add overlays to frame
        frame_with_overlay = self.add_overlays(frame, ear)
        
        return frame_with_overlay, ear, self.is_drowsy(ear)
    
    def add_overlays(self, frame, ear):
        """Add text overlays to frame"""
        frame_copy = frame.copy()
        
        # Add EAR value
        cv2.putText(frame_copy, f'EAR: {ear:.3f}', (30, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Add threshold line
        cv2.putText(frame_copy, f'Threshold: {self.detector.ear_thresh:.3f}', 
                   (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add status
        if ear < self.detector.ear_thresh:
            cv2.putText(frame_copy, 'DROWSY DETECTED!', (100, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            # Add warning rectangle
            cv2.rectangle(frame_copy, (90, 120), (550, 180), (0, 0, 255), 3)
        else:
            cv2.putText(frame_copy, 'ALERT', (30, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame_copy
    
    def is_drowsy(self, ear):
        """Check if current state is drowsy"""
        return ear < self.detector.ear_thresh if self.detector else False
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

def main():
    # Initialize session state
    if 'detector_system' not in st.session_state:
        st.session_state.detector_system = StreamlitDrowsyDetector()
    
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    
    detector_system = st.session_state.detector_system
    
    # Main title
    st.markdown('<h1 class="main-header">😴 Drowsiness Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Detection parameters
        st.subheader("Detection Parameters")
        ear_threshold = st.slider(
            "EAR Threshold", 
            min_value=0.10, 
            max_value=0.35, 
            value=0.18, 
            step=0.01,
            help="Lower values = more sensitive"
        )
        
        wait_time = st.slider(
            "Wait Time (seconds)", 
            min_value=0.5, 
            max_value=3.0, 
            value=1.0, 
            step=0.1,
            help="Time to wait before alerting"
        )
        
        # Camera settings
        st.subheader("Camera Settings")
        camera_source = st.selectbox(
            "Camera Source",
            options=[0, 1, 2],
            index=0,
            help="Select camera index (usually 0 for default camera)"
        )
        
        # Initialize detector button
        if st.button("🔄 Initialize System", type="primary"):
            with st.spinner("Initializing system..."):
                if detector_system.initialize_detector(ear_threshold, wait_time):
                    if detector_system.initialize_camera(camera_source):
                        st.success("✅ System initialized successfully!")
                    else:
                        st.error("❌ Camera initialization failed!")
                else:
                    st.error("❌ Detector initialization failed!")
        
        st.divider()
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Detection"):
                if detector_system.detector is not None:
                    st.session_state.detection_active = True
                    st.success("Detection started!")
                else:
                    st.error("Please initialize system first!")
        
        with col2:
            if st.button("⏹️ Stop Detection"):
                st.session_state.detection_active = False
                st.info("Detection stopped!")
        
        st.divider()
        
        # System info
        st.subheader("📊 System Info")
        uptime = time.time() - detector_system.stats['session_start']
        st.metric("Session Uptime", f"{uptime:.1f}s")
        st.metric("Total Detections", detector_system.stats['total_detections'])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        
        # Video display placeholder
        video_placeholder = st.empty()
        
        # Status display
        status_placeholder = st.empty()
    
    with col2:
        st.subheader("📈 Real-time Metrics")
        
        # Metrics placeholders
        ear_metric = st.empty()
        threshold_metric = st.empty()
        status_metric = st.empty()
        
        st.subheader("ℹ️ Information")
        st.info("""
        **How it works:**
        - The system monitors your Eye Aspect Ratio (EAR)
        - When EAR drops below threshold for the wait time, it alerts for drowsiness
        - Green status = Alert, Red status = Drowsy
        
        **Tips:**
        - Ensure good lighting
        - Position camera at eye level
        - Adjust threshold if too sensitive/insensitive
        """)
    
    # Main detection loop
    if st.session_state.detection_active and detector_system.detector is not None:
        try:
            frame, ear, is_drowsy = detector_system.process_frame()
            
            if frame is not None:
                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                
                # Display frame
                video_placeholder.image(img_pil, channels="RGB", use_column_width=True)
                
                # Update metrics
                ear_metric.metric(
                    "Current EAR", 
                    f"{ear:.3f}",
                    delta=f"Threshold: {ear_threshold:.3f}"
                )
                
                threshold_metric.metric(
                    "Status", 
                    "🚨 DROWSY" if is_drowsy else "✅ ALERT",
                    delta="Action Required!" if is_drowsy else "Normal"
                )
                
                # Update status display
                if is_drowsy:
                    status_placeholder.markdown(
                        '<div class="status-alert status-drowsy">🚨 DROWSINESS DETECTED! 🚨</div>', 
                        unsafe_allow_html=True
                    )
                    detector_system.stats['total_detections'] += 1
                    detector_system.stats['last_drowsy_time'] = time.time()
                else:
                    status_placeholder.markdown(
                        '<div class="status-alert status-normal">✅ Driver Alert</div>', 
                        unsafe_allow_html=True
                    )
                
                # Auto-refresh
                time.sleep(0.1)
                st.rerun()
            
        except Exception as e:
            st.error(f"Error during detection: {str(e)}")
            st.session_state.detection_active = False
    
    elif st.session_state.detection_active:
        st.warning("⚠️ System not initialized. Please initialize first.")
        st.session_state.detection_active = False
    
    else:
        # Show placeholder when not active
        video_placeholder.image("https://via.placeholder.com/640x480/cccccc/000000?text=Camera+Feed+Inactive", 
                               use_column_width=True)
        status_placeholder.markdown(
            '<div class="status-alert">System Inactive - Click Start Detection</div>', 
            unsafe_allow_html=True
        )
    
    # Cleanup on app termination
    if st.session_state.get('cleanup_needed', False):
        detector_system.cleanup()

if __name__ == "__main__":
    main()
