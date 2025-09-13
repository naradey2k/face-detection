# Face Detection Analysis - WIDER FACE Dataset

A comprehensive face detection pipeline implementing multiple state-of-the-art methods for the Junior Computer Vision Engineer assignment.

## 🎯 Project Overview

This project evaluates four different face detection methods on a subset of the WIDER FACE dataset:

- **OpenCV Haar Cascade** - Classic machine learning approach
- **OpenCV DNN** - Deep learning with ResNet-based face detector
- **RetinaFace** - State-of-the-art single-stage face detector
- **SCRFD** - Efficient face detection with simplified implementation
- **MTCNN** - Efficient and accurate face detection model

## 🚀 Quick Start

### Option 1: Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd face-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis**
   ```bash
   python3 run_evaluation.py
   ```

### Option 2: Docker

1. **Build and run with Docker**
   ```bash
   docker build -t face-detection .
   docker run face-detection
   ```

## 📁 Project Structure

```
face-detection-analysis/
├── models/
│   ├── opencv_haar.py         # Haar Cascade implementation
│   ├── opencv_dnn.py          # OpenCV DNN implementation
│   ├── retina_face.py         # RetinaFace implementation
│   ├── scrfd.py               # SCRFD implementation
│   ├── mtcnn.py               # MTCNN implementation
├── evaluation.py               # Evaluation metrics and IoU calculation
├── face_detection_analysis.ipynb  # Main analysis notebook
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
└── README.md                  # This file
```

## 📊 Evaluation Results

Based on evaluation of 500 images from WIDER FACE dataset:

| Model | Precision | Recall | F1-Score | Speed (s/img) | Total Time (s) |
|-------|-----------|--------|----------|---------------|----------------|
| OpenCV Haar | 0.5074 | 0.1164 | 0.1891 | 0.0534 | 26.71 |
| OpenCV DNN | 0.9406 | 0.0769 | 0.1424 | 0.0207 | 10.35 |
| RetinaFace | 0.9081 | 0.1905 | 0.3151 | 0.1843 | 92.16 |
| SCRFD | 0.9315 | 0.3820 | 0.5434 | 0.0756 | 37.79 |
| MTCNN | 0.9168 | 0.3519 | 0.5087 | 0.2792 | 139.60 |

### Key Findings:
- **SCRFD** achieved the best balance with highest F1-Score (0.5434) and good recall (0.3820)
- **OpenCV DNN** had the highest precision (0.9406) but lowest recall (0.0769) - very conservative
- **OpenCV Haar** was fastest but had poor accuracy, detecting many non-face objects
- **RetinaFace** showed good precision (0.9081) with moderate recall (0.1905)
- **MTCNN** provided good balance but was the slowest model

## 🔍 Key Findings

### Performance Analysis
- **RetinaFace** typically achieves the highest accuracy with excellent precision and recall
- **OpenCV DNN** provides a good balance between accuracy and speed
- **SCRFD** offers competitive performance with faster inference
- **OpenCV Haar** is fastest but has lower accuracy, especially on challenging cases
- **MTCNN** has the best balance between accuracy and latency
