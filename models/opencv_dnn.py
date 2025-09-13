import cv2
import numpy as np
import os
import urllib.request

class OpenCVDNNDetector():
    def __init__(self):
        self.confidence_threshold = 0.5
        self.net = None
        self.input_size = (300, 300)  # Changed to match the actual model input size
        self.mean = [104, 117, 123]
        self.model_loaded = False
        
    def _download_model_files(self):
        model_dir = "./models/opencv_dnn_models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Use more reliable URLs for the model files
        model_files = {
            'prototxt': {
                'url': 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt',
                'path': os.path.join(model_dir, 'deploy.prototxt')
            },
            'model': {
                'url': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
                'path': os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
            }
        }
        
        for file_type, file_info in model_files.items():
            if not os.path.exists(file_info['path']):
                print(f"Downloading {file_type} file...")
                try:
                    # Add timeout and better error handling
                    urllib.request.urlretrieve(file_info['url'], file_info['path'])
                    print(f"Downloaded {file_info['path']}")
                    
                    # Verify file was downloaded and has content
                    if os.path.getsize(file_info['path']) == 0:
                        os.remove(file_info['path'])
                        raise Exception(f"Downloaded file is empty: {file_info['path']}")
                        
                except Exception as e:
                    print(f"Error downloading {file_type}: {e}")
                    # Try alternative URLs if primary fails
                    if file_type == 'prototxt':
                        alt_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
                        try:
                            print(f"Trying alternative URL for prototxt...")
                            urllib.request.urlretrieve(alt_url, file_info['path'])
                            print(f"Downloaded {file_info['path']} from alternative URL")
                        except Exception as e2:
                            print(f"Alternative URL also failed: {e2}")
                            raise e
                    else:
                        raise e
        
        return model_files['prototxt']['path'], model_files['model']['path']
    
    def load_model(self):
        try:
            prototxt_path, model_path = self._download_model_files()
            
            # Verify files exist and have content
            if not os.path.exists(prototxt_path) or os.path.getsize(prototxt_path) == 0:
                raise Exception(f"Prototxt file is missing or empty: {prototxt_path}")
            if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                raise Exception(f"Model file is missing or empty: {model_path}")
            
            print(f"Loading OpenCV DNN model from:")
            print(f"  Prototxt: {prototxt_path}")
            print(f"  Model: {model_path}")
            
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            
            if self.net.empty():
                raise Exception("Failed to load model - network is empty")
                
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            self.model_loaded = True
            print("OpenCV DNN model loaded successfully")
            
        except Exception as e:
            print(f"Error loading OpenCV DNN model: {e}")
            self.model_loaded = False
            raise
            
    def detect_faces(self, image_path):
        if not self.model_loaded:
            self.load_model()
            
        image = cv2.imread(image_path)
        if image is None:
            return []
            
        h, w = image.shape[:2]
        
        blob = cv2.dnn.blobFromImage(
            image, 1.0, self.input_size, self.mean, swapRB=False, crop=False
        )
        self.net.setInput(blob)
        
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x1, y1 = box.astype(int)
                
                width = x1 - x
                height = y1 - y
                
                if width > 0 and height > 0:
                    detection = {
                        'bbox': (x, y, width, height),
                        'confidence': float(confidence),
                    }
                    faces.append(detection)
        
        return faces