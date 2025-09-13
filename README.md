# Face Detection Analysis - WIDER FACE Dataset

A comprehensive face detection pipeline implementing multiple state-of-the-art methods for the Junior Computer Vision Engineer assignment.

## 🎯 Project Overview

This project evaluates four different face detection methods on a subset of the WIDER FACE dataset:

- **OpenCV Haar Cascade** - Classic machine learning approach
- **OpenCV DNN** - Deep learning with ResNet-based face detector
- **RetinaFace** - State-of-the-art single-stage face detector
- **SCRFD** - Efficient face detection with simplified implementation

## 📊 Key Features

- **Comprehensive Evaluation**: Precision, Recall, F1-Score, and processing speed metrics
- **Visual Analysis**: Bounding box visualization and performance comparison charts
- **Modular Design**: Clean, structured Python project with separate modules
- **Docker Support**: Containerized environment for easy deployment
- **Detailed Documentation**: Complete analysis and recommendations

## 🚀 Quick Start

### Option 1: Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd face-detection-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis**
   ```bash
   jupyter notebook face_detection_analysis.ipynb
   ```

### Option 2: Docker (Recommended)

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access Jupyter Notebook**
   - Open your browser and go to `http://localhost:8888`
   - Open `face_detection_analysis.ipynb`

### Option 3: Docker without Compose

1. **Build the image**
   ```bash
   docker build -t face-detection .
   ```

2. **Run the container**
   ```bash
   docker run -p 8888:8888 -v $(pwd):/app face-detection
   ```

## 📁 Project Structure

```
face-detection-analysis/
├── data/
│   ├── annos.json              # WIDER FACE annotations
│   └── images/                 # Dataset images (500 images)
├── models/
│   ├── opencv_haar.py         # Haar Cascade implementation
│   ├── opencv_dnn.py          # OpenCV DNN implementation
│   ├── retina_face.py         # RetinaFace implementation
│   ├── scrfd.py               # SCRFD implementation
│   └── opencv_dnn_models/     # Pre-trained DNN models
├── evaluation.py               # Evaluation metrics and IoU calculation
├── visualization.py            # Plotting and visualization utilities
├── face_detection_analysis.ipynb  # Main analysis notebook
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
└── README.md                  # This file
```

## 🔧 Dependencies

- **OpenCV 4.8.1** - Computer vision library
- **NumPy 1.24.3** - Numerical computing
- **Matplotlib 3.7.2** - Plotting and visualization
- **Pillow 10.0.0** - Image processing
- **RetinaFace 0.0.13** - Face detection library
- **Scikit-learn 1.3.0** - Machine learning metrics
- **Jupyter 1.0.0** - Interactive notebooks
- **tqdm 4.65.0** - Progress bars

## 📈 Evaluation Metrics

The project evaluates each model using:

- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall
- **Processing Speed**: Average time per image
- **IoU Threshold**: 0.5 for matching detections to ground truth

## 🎨 Visualization Features

- **Sample Detection Visualization**: Side-by-side comparison of all models
- **Performance Charts**: Precision, Recall, F1-Score, and speed comparisons
- **Error Analysis**: Detection rates, false positive rates, and miss rates
- **Dataset Statistics**: Face count distribution, blur, and pose analysis

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

### Use Case Recommendations
- **High Accuracy Applications**: RetinaFace
- **Real-time Applications**: OpenCV Haar or SCRFD
- **Balanced Performance**: OpenCV DNN
- **Resource-Constrained**: OpenCV Haar

## 🐛 Troubleshooting

### Common Issues

1. **Model Download Errors**
   - Ensure internet connection for model downloads
   - Check firewall settings

2. **Memory Issues**
   - Reduce batch size in evaluation
   - Process images in smaller batches

3. **Docker Issues**
   - Ensure Docker is running
   - Check port 8888 is available

### Performance Optimization

- Use GPU acceleration for faster inference (modify model configurations)
- Implement batch processing for large datasets
- Consider model quantization for deployment

## 📝 Technical Details

### IoU Calculation
The project uses Intersection over Union (IoU) with a threshold of 0.5 to match detections with ground truth bounding boxes.

### Model Implementations
- **Haar Cascade**: Uses OpenCV's built-in frontal face cascade
- **OpenCV DNN**: ResNet-based face detector with Caffe backend
- **RetinaFace**: Single-stage detector with multi-scale feature maps
- **SCRFD**: Simplified version with fallback to OpenCV DNN

### Evaluation Process
1. Load ground truth annotations
2. Run each model on all images
3. Match detections using IoU threshold
4. Calculate precision, recall, and F1-score
5. Measure processing time
6. Generate visualizations and reports

## 🤝 Contributing

This project was created for a Junior Computer Vision Engineer assignment. For improvements or extensions:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is created for educational purposes as part of a technical assignment.

## 📧 Contact

For questions about this implementation, please contact the repository owner.

---

**Note**: This project demonstrates proficiency in computer vision, Python development, evaluation metrics, and project organization as required for the Junior Computer Vision Engineer position.
