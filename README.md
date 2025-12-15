# DriverDrowsyApp

A real-time driver drowsiness detection system using YOLOv8 for face localization combined with MediaPipe Face Mesh for advanced feature-based temporal analysis. The system uses Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and PERCLOS metrics with personalized calibration to accurately detect driver fatigue.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Detection Application](#running-the-detection-application)
  - [Calibration Process](#calibration-process)
  - [Training Your Own Model](#training-your-own-model)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

**DriverDrowsyApp** is an advanced computer vision application designed to detect driver drowsiness in real-time using a webcam feed. Unlike simple classification approaches, this system employs a **feature-based temporal analysis pipeline**:

1. **YOLOv8** localizes the driver's face
2. **MediaPipe Face Mesh** extracts 468 facial landmarks
3. **EAR (Eye Aspect Ratio)** monitors eye openness
4. **MAR (Mouth Aspect Ratio)** detects yawning
5. **PERCLOS** tracks percentage of eye closure over time
6. **Dynamic Calibration** personalizes thresholds for each user

The application is particularly useful for:
- Fleet management systems
- Personal vehicle safety enhancement
- Driver monitoring systems
- Research on driver fatigue detection

---

## Features

- **Feature-Based Detection**: Uses EAR, MAR, and PERCLOS instead of simple classification
- **Dynamic Calibration**: 3-second startup calibration personalizes thresholds to your face
- **Real-time Detection**: Processes live webcam feed with minimal latency
- **Visual Alerts**: Displays prominent "DROWSY!" warning when fatigue is detected
- **Comprehensive Metrics Display**:
  - EAR (Eye Aspect Ratio) with threshold
  - MAR (Mouth Aspect Ratio) with threshold
  - PERCLOS percentage
  - Blink counter
  - Yawn counter
  - Calibrated baseline values
- **Color-coded Indicators**: 
  - 🟢 Green: Normal/Alert state
  - 🔴 Red: Drowsy state detected
  - 🟡 Yellow: Calibrating
- **Temporal Analysis**: Tracks patterns over 30-second sliding window
- **GPU Acceleration**: Supports CUDA-enabled GPUs for faster inference

---

## How It Works

### Detection Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Webcam    │───▶│   YOLOv8     │───▶│  MediaPipe      │
│   Frame     │    │ Face Detect  │    │  Face Mesh      │
└─────────────┘    └──────────────┘    └─────────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Fatigue    │◀───│   Temporal   │◀───│  EAR & MAR      │
│  Decision   │    │   Analysis   │    │  Calculation    │
└─────────────┘    └──────────────┘    └─────────────────┘
```

### Key Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **EAR** | Eye Aspect Ratio - measures eye openness | `(‖P2-P6‖ + ‖P3-P5‖) / (2 × ‖P1-P4‖)` |
| **MAR** | Mouth Aspect Ratio - detects yawning | `(vertical distances) / (horizontal distance)` |
| **PERCLOS** | Percentage of Eye Closure - tracks fatigue over time | `(closed frames / total frames) × 100` |

### Fatigue Detection Criteria

The system triggers a **DROWSY** alert when ANY of these conditions are met:
- **PERCLOS > 35%**: Eyes closed more than 35% of the time over 30 seconds
- **Prolonged Eye Closure**: Eyes closed for 15+ consecutive frames
- **Active Yawning**: Mouth open (MAR > 0.55) for 15+ consecutive frames

---

## Project Structure

```
DriverDrowsyApp/
│
├── app_drowsiness.py          # Main application for real-time detection
├── train_yolov8.py            # Training script for custom model
├── requirements.txxt.txt      # Python dependencies
├── best.pt                    # Pre-trained model weights (backup)
├── yolov8n.pt                 # YOLOv8 nano base model
│
└── runs/
    ├── detect/
    │   └── val/               # Validation results
    │
    └── drowsiness_model_v8/   # Trained model directory
        ├── args.yaml          # Training configuration
        ├── results.csv        # Training metrics log
        └── weights/
            ├── best.pt        # Best performing model weights
            └── last.pt        # Final epoch model weights
```

---

## Requirements

### System Requirements

- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 - 3.12 (MediaPipe requires Python ≤3.12)
- **Webcam**: Built-in or external USB camera
- **GPU** (Optional but recommended): NVIDIA GPU with CUDA support

### Python Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| ultralytics | ≥8.3.237 | YOLOv8 framework |
| mediapipe | ≥0.10.9 | Face mesh landmark detection |
| scipy | ≥1.11.0 | Distance calculations for EAR/MAR |
| torch | ≥2.1.0 | PyTorch deep learning library |
| torchvision | ≥0.16.0 | Computer vision utilities |
| opencv-python | ≥4.7.0 | Image processing library |
| numpy | ≥1.25.0 | Numerical computing |
| tqdm | ≥4.65.0 | Progress bar utilities |

> ⚠️ **Note**: MediaPipe does not support Python 3.13+. Use Python 3.12 or earlier.

---

## Installation

### 1. Clone or Download the Repository

```bash
# If using git
git clone <repository-url>
cd DriverDrowsyApp

# Or extract the downloaded ZIP file
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install from requirements file
pip install -r requirements.txxt.txt

# Or install individually
pip install ultralytics>=8.3.237
pip install mediapipe>=0.10.9
pip install scipy>=1.11.0
pip install torch torchvision torchaudio
pip install opencv-python>=4.7.0
pip install numpy>=1.25.0
pip install tqdm>=4.65.0
```

### 4. For GPU Support (NVIDIA)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

---

## Usage

### Running the Detection Application

1. **Update the Model Path** (if necessary)
   
   Open `app_drowsiness.py` and update the `MODEL_PATH` variable to point to your model:
   
   ```python
   MODEL_PATH = r"path/to/your/DriverDrowsyApp/runs/drowsiness_model_v8/weights/best.pt"
   ```

2. **Run the Application**
   
   ```bash
   python app_drowsiness.py
   ```

3. **Using the Application**
   - The webcam feed will open in a new window
   - **CALIBRATION PHASE** (first 3 seconds):
     - Keep your eyes **WIDE OPEN** and look at the camera
     - A progress bar will show calibration status
     - The system calculates your personal EAR baseline
   - **DETECTION PHASE** (after calibration):
     - The system monitors your eyes and mouth in real-time
     - Metrics panel shows EAR, MAR, PERCLOS, blinks, and yawns
     - **Green bounding box**: Normal/Alert state
     - **Red bounding box + "DROWSY!" text**: Fatigue detected
   - Press **'q'** to quit the application

### Calibration Process

The calibration phase is crucial for accurate detection:

1. **Why Calibration?**
   - Everyone's eyes have different natural EAR values
   - Generic thresholds cause false positives for some users
   - Calibration creates personalized thresholds

2. **During Calibration (3 seconds)**
   - Keep eyes naturally open (don't force them wide)
   - Look directly at the camera
   - Avoid blinking if possible
   - Stay still

3. **What Happens**
   - System measures your average EAR over 90 frames
   - Calculates your `OPEN_EAR_BASELINE`
   - Sets `EYE_AR_THRESH = BASELINE × 0.78`
   - This means eyes are "closed" when EAR drops below 78% of your normal

4. **After Calibration**
   - Your personalized baseline is displayed
   - Detection becomes active with calibrated thresholds
   - Metrics panel shows current values vs thresholds

### Training Your Own Model

If you want to train the model on your own dataset:

1. **Prepare Your Dataset**
   - Organize your dataset in YOLO format
   - Create a `data.yaml` file with paths and class names

2. **Update Training Configuration**
   
   Edit `train_yolov8.py`:
   ```python
   DATA_YAML = r"path/to/your/data.yaml"
   PROJECT_DIR = r"path/to/save/results"
   ```

3. **Run Training**
   
   ```bash
   python train_yolov8.py
   ```

4. **Training Parameters**

   | Parameter | Default | Description |
   |-----------|---------|-------------|
   | EPOCHS | 50 | Number of training epochs |
   | BATCH_SIZE | 16 | Batch size for training |
   | IMGSZ | 640 | Input image size |
   | PATIENCE | 10 | Early stopping patience |

---

## Model Performance

The included pre-trained model was trained for **50 epochs** and achieved excellent performance:

### Final Metrics (Epoch 50)

| Metric | Value |
|--------|-------|
| **Precision** | 96.65% |
| **Recall** | 96.02% |
| **mAP@50** | 98.65% |
| **mAP@50-95** | 87.88% |

### Training Progress

The model showed consistent improvement throughout training:

| Epoch | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| 1 | 68.30% | 62.27% | 71.55% | 50.00% |
| 10 | 94.82% | 93.42% | 97.35% | 79.21% |
| 25 | 94.20% | 95.58% | 98.47% | 84.55% |
| 50 | 96.65% | 96.02% | 98.65% | 87.88% |

### Loss Curves

Training losses decreased steadily:
- **Box Loss**: 1.24 → 0.66
- **Classification Loss**: 1.92 → 0.39
- **DFL Loss**: 1.38 → 0.99

---

## Configuration

### Application Settings (`app_drowsiness.py`)

```python
# Model Configuration
MODEL_PATH = "path/to/best.pt"        # Path to trained model weights

# Window Settings
WINDOW_NAME = "Drowsiness Detection"  # Display window title
WINDOW_WIDTH = 640                    # Window width in pixels
WINDOW_HEIGHT = 480                   # Window height in pixels

# Detection Settings
CONFIDENCE_THRESHOLD = 0.5            # Minimum confidence for face detection (0.0-1.0)

# Feature-Based Thresholds
EYE_AR_THRESH = 0.25                  # Initial EAR threshold (calibrated dynamically)
EYE_AR_CONSEC_FRAMES = 15             # Frames for sustained eye closure
MAR_THRESH = 0.55                     # Mouth aspect ratio for yawn detection
MAR_CONSEC_FRAMES = 15                # Frames for sustained yawn
PERCLOS_WINDOW_SIZE = 900             # Sliding window (30 sec at 30 FPS)
PERCLOS_THRESH = 35.0                 # PERCLOS percentage threshold

# Calibration Settings
CALIBRATION_FRAMES_TOTAL = 90         # Calibration duration (~3 sec at 30 FPS)
```

### Tuning Thresholds

If you experience issues, adjust these values:

| Issue | Solution |
|-------|----------|
| Too many false positives | Increase `PERCLOS_THRESH` (e.g., 40-50%) |
| Missing real drowsiness | Decrease `PERCLOS_THRESH` (e.g., 25-30%) |
| Yawn detection too sensitive | Increase `MAR_THRESH` (e.g., 0.6-0.7) |
| Not detecting yawns | Decrease `MAR_THRESH` (e.g., 0.4-0.5) |
| Calibration too short | Increase `CALIBRATION_FRAMES_TOTAL` |

### Training Settings (`train_yolov8.py`)

```python
# Dataset Configuration
DATA_YAML = "path/to/data.yaml"       # Dataset configuration file

# Training Hyperparameters
EPOCHS = 50                           # Number of training epochs
BATCH_SIZE = 16                       # Batch size
IMGSZ = 640                           # Input image size
PATIENCE = 10                         # Early stopping patience

# Output Configuration
PROJECT_DIR = "path/to/runs"          # Directory to save results
RUN_NAME = "drowsiness_model_v8"      # Name for this training run
```

---

## Technical Details

### Detection Architecture

The system uses a two-stage approach:

1. **Stage 1: Face Localization (YOLOv8)**
   - Model: YOLOv8n (nano) - lightweight and fast
   - Task: Face/object detection
   - Input Size: 640×640 pixels
   - Output: Bounding box coordinates

2. **Stage 2: Landmark Detection (MediaPipe)**
   - Model: MediaPipe Face Mesh
   - Landmarks: 468 facial points
   - Key landmarks used:
     - Left Eye: indices [362, 385, 387, 263, 373, 380]
     - Right Eye: indices [33, 160, 158, 133, 153, 144]
     - Mouth: indices [61, 291, 39, 181, 0, 17, 269, 405]

### EAR Calculation

The Eye Aspect Ratio is calculated using 6 landmarks per eye:

```
       P2    P3
        \  /
    P1 -------- P4
        /  \
       P6    P5

EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)
```

- **High EAR** (~0.3-0.4): Eyes open
- **Low EAR** (<0.2): Eyes closed
- **Calibrated threshold**: 78% of your personal baseline

### PERCLOS Algorithm

PERCLOS (Percentage of Eye Closure) is a validated fatigue metric:

```python
PERCLOS = (frames_with_eyes_closed / total_frames_in_window) × 100
```

- Window size: 900 frames (~30 seconds at 30 FPS)
- Uses a sliding window (deque) for efficiency
- Threshold: 35% (eyes closed >35% of time = fatigued)

### Model Architecture (YOLOv8)

### Training Configuration

Key hyperparameters used during training:

| Parameter | Value |
|-----------|-------|
| Optimizer | Auto |
| Initial Learning Rate | 0.01 |
| Final Learning Rate | 0.01 |
| Momentum | 0.937 |
| Weight Decay | 0.0005 |
| Warmup Epochs | 3.0 |
| IoU Threshold | 0.7 |
| Mixed Precision (AMP) | Enabled |

### Data Augmentation

The model was trained with standard YOLOv8 augmentations:

| Augmentation | Value |
|--------------|-------|
| HSV Hue | 0.015 |
| HSV Saturation | 0.7 |
| HSV Value | 0.4 |
| Translation | 0.1 |
| Scale | 0.5 |
| Horizontal Flip | 0.5 |
| Mosaic | 1.0 |

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Webcam Not Detected

```
[ERROR] Could not read from webcam.
```

**Solutions:**
- Ensure your webcam is properly connected
- Try changing the camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`
- Check if another application is using the webcam
- Update webcam drivers

#### 2. Model File Not Found

**Solutions:**
- Verify the `MODEL_PATH` is correct
- Ensure `best.pt` exists in the specified directory
- Use absolute paths to avoid path issues

#### 3. MediaPipe Installation Error (Python 3.13+)

```
ERROR: No matching distribution found for mediapipe
```

**Solutions:**
- MediaPipe requires Python 3.8-3.12
- Install Python 3.12: `winget install Python.Python.3.12`
- Create a new virtual environment with Python 3.12:
  ```bash
  py -3.12 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install mediapipe scipy ultralytics opencv-python
  ```

#### 4. Constant False Positives (Always shows DROWSY)

**Solutions:**
- Ensure you keep eyes **OPEN** during calibration
- Check lighting conditions (low light affects EAR calculation)
- Increase `PERCLOS_THRESH` to 40-50%
- The calibration baseline may be too low - restart the app

#### 5. Not Detecting Drowsiness

**Solutions:**
- Check if calibration completed successfully
- Decrease `PERCLOS_THRESH` to 25-30%
- Ensure face is clearly visible and well-lit
- Check the EAR values in the metrics panel

#### 6. CUDA/GPU Errors

**Solutions:**
- Verify NVIDIA drivers are installed
- Check CUDA version compatibility with PyTorch
- Fall back to CPU: The application will automatically use CPU if GPU is unavailable

#### 7. Slow Performance

**Solutions:**
- Enable GPU acceleration if available
- Reduce `WINDOW_WIDTH` and `WINDOW_HEIGHT`
- Increase `CONFIDENCE_THRESHOLD` to filter more detections

#### 8. Import Errors

**Solutions:**
- Ensure all dependencies are installed: `pip install -r requirements.txxt.txt`
- Check Python version compatibility (3.8-3.12)
- Reinstall packages: `pip install --upgrade mediapipe scipy ultralytics`

---

## Future Improvements

Potential enhancements for the project:

- [ ] Audio alerts for drowsiness detection
- [ ] Logging and statistics tracking
- [ ] Integration with vehicle systems
- [ ] Mobile application version
- [ ] Support for multiple cameras
- [ ] Driver identification features
- [ ] Cloud-based monitoring dashboard
- [ ] Head pose estimation for distraction detection
- [ ] Configurable calibration duration
- [ ] Save/load calibration profiles

---

## License

This project is provided for educational and research purposes. Please ensure compliance with local regulations regarding driver monitoring systems.

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [MediaPipe](https://mediapipe.dev/) - Face mesh landmark detection
- [OpenCV](https://opencv.org/) - Computer vision library
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [SciPy](https://scipy.org/) - Scientific computing library

---

## References

- Soukupová, T., & Čech, J. (2016). "Real-Time Eye Blink Detection using Facial Landmarks"
- PERCLOS metric: Dinges, D. F., & Grace, R. (1998). "PERCLOS: A valid psychophysiological measure of alertness"

---

## Contact

For questions, issues, or contributions, please open an issue in the repository or contact the project maintainers.

---

*Last Updated: December 2025*
