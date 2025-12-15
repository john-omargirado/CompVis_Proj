# DriverDrowsyApp

A real-time driver drowsiness detection system using YOLOv8 and computer vision to enhance road safety by alerting drivers when signs of drowsiness are detected.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Detection Application](#running-the-detection-application)
  - [Training Your Own Model](#training-your-own-model)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

**DriverDrowsyApp** is a computer vision application designed to detect driver drowsiness in real-time using a webcam feed. The system leverages the power of YOLOv8 (You Only Look Once version 8) object detection model to identify visual signs of drowsiness in drivers and provides immediate visual alerts.

The application is particularly useful for:
- Fleet management systems
- Personal vehicle safety enhancement
- Driver monitoring systems
- Research on driver fatigue detection

---

## Features

- **Real-time Detection**: Processes live webcam feed with minimal latency
- **Visual Alerts**: Displays prominent "DROWSY!" warning when drowsiness is detected
- **Bounding Box Visualization**: Shows detection regions with confidence scores
- **Color-coded Indicators**: 
  - 🟢 Green: Normal/Alert state
  - 🔴 Red: Drowsy state detected
- **Configurable Confidence Threshold**: Adjustable sensitivity for detection
- **GPU Acceleration**: Supports CUDA-enabled GPUs for faster inference
- **Pre-trained Model**: Includes a trained model ready for immediate use

---

## Project Structure

```
DriverDrowsyApp/
│
├── app_drowsiness.py          # Main application for real-time detection
├── train_yolov8.py            # Training script for custom model
├── requirements.txt      # Python dependencies
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
- **Python**: 3.8 or higher
- **Webcam**: Built-in or external USB camera
- **GPU** (Optional but recommended): NVIDIA GPU with CUDA support

### Python Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| ultralytics | ≥8.3.237 | YOLOv8 framework |
| torch | ≥2.1.0 | PyTorch deep learning library |
| torchvision | ≥0.16.0 | Computer vision utilities |
| opencv-python | ≥4.7.0 | Image processing library |
| numpy | ≥1.25.0 | Numerical computing |
| tqdm | ≥4.65.0 | Progress bar utilities |

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
   - Position yourself in front of the camera
   - The system will automatically detect and classify your state
   - **Green bounding box**: Normal/Alert state
   - **Red bounding box + "DROWSY!" text**: Drowsiness detected
   - Press **'q'** to quit the application

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
CONFIDENCE_THRESHOLD = 0.5            # Minimum confidence for detection (0.0-1.0)
```

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

### Model Architecture

- **Base Model**: YOLOv8n (nano) - lightweight and fast
- **Task**: Object Detection
- **Input Size**: 640×640 pixels
- **Classes**: Drowsy, Normal (presumed based on detection logic)

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

#### 3. CUDA/GPU Errors

**Solutions:**
- Verify NVIDIA drivers are installed
- Check CUDA version compatibility with PyTorch
- Fall back to CPU: The application will automatically use CPU if GPU is unavailable

#### 4. Slow Performance

**Solutions:**
- Enable GPU acceleration if available
- Reduce `WINDOW_WIDTH` and `WINDOW_HEIGHT`
- Increase `CONFIDENCE_THRESHOLD` to filter more detections

#### 5. Import Errors

**Solutions:**
- Ensure all dependencies are installed: `pip install -r requirements.txxt.txt`
- Check Python version compatibility (3.8+)
- Reinstall ultralytics: `pip install --upgrade ultralytics`

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

---

## License

This project is provided for educational and research purposes. Please ensure compliance with local regulations regarding driver monitoring systems.

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [PyTorch](https://pytorch.org/) - Deep learning framework

---

## Contact

For questions, issues, or contributions, please open an issue in the repository or contact the project maintainers.

---

*Last Updated: December 2025*

