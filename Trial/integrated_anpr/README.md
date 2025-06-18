# Integrated ANPR System

This is an integrated Automatic Number Plate Recognition (ANPR) system that combines YOLOv8 for vehicle detection and EasyOCR for license plate text recognition, with a modern PyQt-based GUI.

## Features

- Real-time vehicle detection using YOLOv8
- License plate text recognition using EasyOCR
- Support for multiple input sources:
  - Live camera feed
  - Video files
  - Images
- Modern and user-friendly GUI
- Real-time display of detection results

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLOv8 model:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Usage

1. Run the application:
```bash
python gui/main_window.py
```

2. Select input source (Camera/Video File/Image)
3. Click 'Start' to begin detection
4. Use 'Stop' to end detection
5. Use 'Browse File' to select video or image files

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for better performance)
- Webcam (for live detection)

## License

MIT License 