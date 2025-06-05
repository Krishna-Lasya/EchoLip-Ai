import cv2
import numpy as np
import os
import logging
from tqdm import tqdm
import mediapipe as mp
from mtcnn import MTCNN
import dlib
import imutils
from imutils import face_utils
import argparse
from pathlib import Path
import json
import time
import sys
import warnings

# Set TensorFlow environment variables BEFORE importing TensorFlow-dependent libraries
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage (optional)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
import tensorflow as tf
if tf.__version__:
    tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow info messages

# Check Python version compatibility
if sys.version_info < (3, 8):
    raise RuntimeError("This script requires Python 3.8 or higher")
    
print(f"Python version: {sys.version}")
print(f"Running on Python {sys.version_info.major}.{sys.version_info.minor} - Compatible!")
print(f"TensorFlow version: {tf.__version__ if 'tf' in locals() else 'Not installed'}")
print(f"TF_ENABLE_ONEDNN_OPTS: {os.environ.get('TF_ENABLE_ONEDNN_OPTS', 'Not set')}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Complete Video Processing Pipeline for Face Extraction and Processing
    
    Pipeline Steps:
    1. Extract Frames (25-30 FPS)
    2. Face Detection (MTCNN/MediaPipe/Dlib)
    3. Face Alignment (68 Landmarks)
    4. Crop Face Region (256×256)
    5. Quality Filter (Blur/Lighting Check)
    """
    
    def __init__(self, detection_method='mtcnn', target_fps=25, face_size=256):
        """
        Initialize Video Processor
        
        Args:
            detection_method: 'mtcnn', 'mediapipe', or 'dlib'
            target_fps: Target frame rate for extraction
            face_size: Output face image size (square)
        """
        self.detection_method = detection_method
        self.target_fps = target_fps
        self.face_size = face_size
        
        # Initialize face detector based on method
        self._init_face_detector()
        
        # Initialize face landmark predictor (dlib)
        try:
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            logger.info("Dlib landmark predictor loaded successfully")
        except:
            logger.warning("Dlib predictor not found. Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            self.predictor = None
    
    def _init_face_detector(self):
        """Initialize the face detection model"""
        if self.detection_method == 'mtcnn':
            self.detector = MTCNN(min_face_size=40, scale_factor=0.709, steps_threshold=[0.6, 0.7, 0.7])
            logger.info("MTCNN detector initialized")
            
        elif self.detection_method == 'mediapipe':
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            logger.info("MediaPipe detector initialized")
            
        elif self.detection_method == 'dlib':
            self.detector = dlib.get_frontal_face_detector()
            logger.info("Dlib detector initialized")
    
    def extract_frames(self, video_path, output_dir):
        """
        Step 1: Extract frames from video at target FPS
        
        Input: Video file path
        Output: Individual frame images
        """
        logger.info(f"Starting frame extraction from: {video_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"Video FPS: {fps}, Total Frames: {total_frames}, Duration: {duration:.2f}s")
        
        # Calculate frame skip ratio
        frame_skip = max(1, int(fps / self.target_fps))
        
        extracted_frames = []
        frame_count = 0
        saved_count = 0
        
        pbar = tqdm(total=total_frames, desc="Extracting frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at target FPS
            if frame_count % frame_skip == 0:
                frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_frames.append(frame_path)
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        logger.info(f"Extracted {saved_count} frames at ~{self.target_fps} FPS")
        return extracted_frames
    
    def detect_faces(self, frame):
        """
        Step 2: Detect faces in frame using selected method
        
        Input: Frame image
        Output: Face bounding boxes and confidence scores
        """
        faces = []
        
        if self.detection_method == 'mtcnn':
            result = self.detector.detect_faces(frame)
            for face in result:
                x, y, w, h = face['box']
                confidence = face['confidence']
                if confidence > 0.7:  # Confidence threshold
                    faces.append({
                        'bbox': (x, y, w, h),
                        'confidence': confidence,
                        'keypoints': face.get('keypoints', {})
                    })
        
        elif self.detection_method == 'mediapipe':
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb_frame)
            
            if results.detections:
                h, w, _ = frame.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    faces.append({
                        'bbox': (x, y, width, height),
                        'confidence': detection.score[0],
                        'keypoints': {}
                    })
        
        elif self.detection_method == 'dlib':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = self.detector(gray)
            
            for det in dets:
                x, y, w, h = det.left(), det.top(), det.width(), det.height()
                faces.append({
                    'bbox': (x, y, w, h),
                    'confidence': 1.0,  # Dlib doesn't provide confidence
                    'keypoints': {}
                })
        
        return faces
    
    def get_face_landmarks(self, frame, bbox):
        """
        Step 3: Get 68 facial landmarks for face alignment
        
        Input: Frame and face bounding box
        Output: 68 facial landmark points
        """
        if self.predictor is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = bbox
        rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Get landmarks
        landmarks = self.predictor(gray, rect)
        landmarks = face_utils.shape_to_np(landmarks)
        
        return landmarks
    
    def align_face(self, frame, landmarks):
        """
        Step 3: Align face using eye landmarks
        
        Input: Frame and facial landmarks
        Output: Aligned face image
        """
        try:
            if landmarks is None or len(landmarks) < 68:
                return frame
            
            # Get eye centers
            left_eye = landmarks[36:42]  # Left eye landmarks
            right_eye = landmarks[42:48]  # Right eye landmarks
            
            # Calculate eye centers with proper type conversion
            left_eye_center = left_eye.mean(axis=0).astype(np.float32)
            right_eye_center = right_eye.mean(axis=0).astype(np.float32)
            
            # Calculate angle
            dy = float(right_eye_center[1] - left_eye_center[1])
            dx = float(right_eye_center[0] - left_eye_center[0])
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Get center point between eyes - convert to tuple of floats
            eye_center = (
                float((left_eye_center[0] + right_eye_center[0]) / 2),
                float((left_eye_center[1] + right_eye_center[1]) / 2)
            )
            
            # Rotate image
            M = cv2.getRotationMatrix2D(eye_center, float(angle), 1.0)
            aligned = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            
            return aligned
            
        except Exception as e:
            logger.warning(f"Face alignment failed: {str(e)}, returning original frame")
            return frame
    
    def crop_face(self, frame, bbox, padding=0.3):
        """
        Step 4: Crop face region to target size
        
        Input: Frame and face bounding box
        Output: Cropped face image (256x256)
        """
        try:
            x, y, w, h = bbox
            
            # Ensure bbox values are integers
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Add padding
            pad_w = int(w * padding)
            pad_h = int(h * padding)
            
            # Calculate crop coordinates
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(frame.shape[1], x + w + pad_w)
            y2 = min(frame.shape[0], y + h + pad_h)
            
            # Ensure coordinates are valid
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid crop coordinates: ({x1},{y1}) to ({x2},{y2})")
                return None
            
            # Crop face
            face_crop = frame[y1:y2, x1:x2]
            
            # Resize to target size
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, (self.face_size, self.face_size))
                return face_resized
            
            return None
            
        except Exception as e:
            logger.warning(f"Face cropping failed: {str(e)}")
            return None
    
    def quality_filter(self, face_image, blur_threshold=100, brightness_range=(50, 200)):
        """
        Step 5: Filter faces based on quality metrics
        
        Input: Face image
        Output: Quality score and pass/fail decision
        """
        if face_image is None:
            return False, 0
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur detection using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Brightness check
        brightness = np.mean(gray)
        
        # 3. Contrast check
        contrast = gray.std()
        
        # Quality decision
        is_good_quality = (
            blur_score > blur_threshold and
            brightness_range[0] < brightness < brightness_range[1] and
            contrast > 30  # Minimum contrast threshold
        )
        
        quality_score = (blur_score / 200) * (contrast / 100) * (
            1 - abs(brightness - 125) / 125)  # Normalized quality score
        
        return is_good_quality, quality_score
    
    def process_video(self, video_path, output_dir):
        """
        Complete video processing pipeline
        
        Input: Video file path
        Output: Processed face images and metadata
        """
        logger.info(f"Starting video processing pipeline for: {video_path}")
        start_time = time.time()
        
        # Create output directories
        frames_dir = os.path.join(output_dir, "frames")
        faces_dir = os.path.join(output_dir, "faces")
        os.makedirs(faces_dir, exist_ok=True)
        
        # Step 1: Extract frames
        frame_paths = self.extract_frames(video_path, frames_dir)
        
        # Process each frame
        processed_faces = []
        face_count = 0
        
        logger.info("Processing frames for face detection and extraction...")
        pbar = tqdm(frame_paths, desc="Processing faces")
        
        for frame_path in pbar:
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # Step 2: Detect faces
            faces = self.detect_faces(frame)
            
            for i, face_data in enumerate(faces):
                bbox = face_data['bbox']
                confidence = face_data['confidence']
                
                # Step 3: Get landmarks and align face
                landmarks = self.get_face_landmarks(frame, bbox)
                if landmarks is not None:
                    aligned_frame = self.align_face(frame, landmarks)
                else:
                    aligned_frame = frame
                
                # Step 4: Crop face
                face_crop = self.crop_face(aligned_frame, bbox)
                
                if face_crop is not None:
                    # Step 5: Quality filter
                    is_good, quality_score = self.quality_filter(face_crop)
                    
                    if is_good:
                        # Save processed face
                        face_filename = f"face_{face_count:06d}.jpg"
                        face_path = os.path.join(faces_dir, face_filename)
                        cv2.imwrite(face_path, face_crop)
                        
                        # Store metadata
                        processed_faces.append({
                            'face_id': face_count,
                            'source_frame': os.path.basename(frame_path),
                            'bbox': [int(b) for b in bbox],  # Convert to regular Python ints
                            'confidence': float(confidence),  # Convert to regular Python float
                            'quality_score': float(quality_score),
                            'face_path': face_path,
                            'landmarks': landmarks.tolist() if landmarks is not None else None
                        })
                        
                        face_count += 1
            
            pbar.set_postfix({'Faces': face_count})
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "processing_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                'video_path': video_path,
                'detection_method': self.detection_method,
                'target_fps': self.target_fps,
                'face_size': self.face_size,
                'total_frames': len(frame_paths),
                'total_faces': face_count,
                'processing_time': time.time() - start_time,
                'faces': processed_faces
            }, f, indent=2)
        
        logger.info(f"Processing complete! Extracted {face_count} quality faces in {time.time() - start_time:.2f}s")
        return processed_faces, metadata_path

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Video Processing Pipeline for Face Extraction')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--method', choices=['mtcnn', 'mediapipe', 'dlib'], 
                       default='mtcnn', help='Face detection method')
    parser.add_argument('--fps', type=int, default=25, help='Target FPS for frame extraction')
    parser.add_argument('--size', type=int, default=256, help='Output face image size')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = VideoProcessor(
        detection_method=args.method,
        target_fps=args.fps,
        face_size=args.size
    )
    
    # Process video
    try:
        faces, metadata_path = processor.process_video(args.video, args.output)
        print(f"\n✅ Processing completed successfully!")
        print(f"📁 Output directory: {args.output}")
        print(f"👥 Total faces extracted: {len(faces)}")
        print(f"📊 Metadata saved to: {metadata_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

# Function for direct usage with your specific paths
def process_echolip_video():
    """
    Process video for EchoLip-AI project with your specific paths
    """
    # Your specific paths
    video_path = r"E:\sample_input.mp4"
    audio_path = r"E:\translated_audio.wav"  # For future audio processing
    output_dir = r"E:\Minho\EchoLip-AI"
    
    print("🎬 EchoLip-AI Video Processing Pipeline")
    print("=" * 50)
    print(f"📹 Input Video: {video_path}")
    print(f"🎵 Audio File: {audio_path}")
    print(f"📁 Output Directory: {output_dir}")
    print("=" * 50)
    
    # Check if input files exist
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.exists(audio_path):
        print(f"⚠️  Audio file not found: {audio_path}")
        print("Audio processing will be skipped for now")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processor with optimal settings for lip-sync
    processor = VideoProcessor(
        detection_method='mtcnn',  # Best accuracy for lip-sync
        target_fps=25,             # Good balance for lip-sync
        face_size=256              # Standard size for AI models
    )
    
    try:
        print("\n🚀 Starting video processing...")
        faces, metadata_path = processor.process_video(video_path, output_dir)
        
        print(f"\n✅ EchoLip-AI Processing Complete!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"👥 Total faces extracted: {len(faces)}")
        print(f"📊 Processing metadata: {metadata_path}")
        
        # Show output structure
        print(f"\n📂 Output Structure:")
        print(f"   {output_dir}/")
        print(f"   ├── frames/              # Extracted video frames")
        print(f"   ├── faces/               # Processed face images")
        print(f"   └── processing_metadata.json  # Processing details")
        
        return faces, metadata_path
        
    except Exception as e:
        logger.error(f"EchoLip-AI processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Direct execution for EchoLip-AI
        print("🎬 Running EchoLip-AI Video Processing...")
        process_echolip_video()

"""
INSTALLATION GUIDE for Python 3.10:

# Method 1: pip (Recommended for Python 3.10)
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install tqdm==4.65.0
pip install imutils==0.5.4
pip install mediapipe==0.10.3
pip install mtcnn==0.1.1
pip install tensorflow==2.13.0  # Required for MTCNN and MediaPipe

# For dlib (might need special handling on Windows):
pip install dlib==19.24.2
# If dlib fails, try:
# conda install -c conda-forge dlib

# Method 2: conda (Alternative)
conda create -n echolip python=3.10
conda activate echolip
conda install -c conda-forge opencv numpy tqdm imutils dlib tensorflow
pip install mediapipe mtcnn

TENSORFLOW ENVIRONMENT VARIABLES:

# Method 1: Set in Windows Command Prompt (before running Python)
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2
python video_processor.py

# Method 2: Set in PowerShell
$env:TF_ENABLE_ONEDNN_OPTS="0"
$env:TF_CPP_MIN_LOG_LEVEL="2"
python video_processor.py

# Method 3: Set permanently in Windows System Environment Variables
# 1. Right-click 'This PC' → Properties → Advanced System Settings
# 2. Click 'Environment Variables'
# 3. Under 'User variables', click 'New'
# 4. Variable name: TF_ENABLE_ONEDNN_OPTS
# 5. Variable value: 0
# 6. Click OK and restart your terminal

# Method 4: Create a batch file (run_echolip.bat)
@echo off
set TF_ENABLE_ONEDNN_OPTS=0
set TF_CPP_MIN_LOG_LEVEL=2
python video_processor.py
pause

# Method 5: Python .env file (create .env file in same directory)
TF_ENABLE_ONEDNN_OPTS=0
TF_CPP_MIN_LOG_LEVEL=2
# Then: pip install python-dotenv

USAGE EXAMPLES:

# Method 1: Direct execution (easiest for your case)
python video_processor.py

# Method 2: Command line with custom parameters
python video_processor.py --video "E:\sample_input.mp4" --output "E:\Minho\EchoLip-AI" --method mtcnn

# Method 3: In Python script
from video_processor import process_echolip_video
faces, metadata = process_echolip_video()

WHY TF_ENABLE_ONEDNN_OPTS=0 is needed:
- Disables Intel oneDNN optimizations that can cause warnings/errors
- Prevents TensorFlow from using CPU optimizations that might be incompatible
- Ensures stable operation with MTCNN and MediaPipe
- Reduces verbose TensorFlow logging

COMPATIBILITY NOTES for Python 3.10:
✅ All libraries are compatible with Python 3.10
✅ Tested on Windows 10/11 with Python 3.10.x
✅ Uses raw strings (r"") for Windows paths
✅ Automatic directory creation
✅ Error handling for missing files
✅ TensorFlow environment variables set automatically in code
"""