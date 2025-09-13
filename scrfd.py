import cv2
import numpy as np
from PIL import Image
from scrfd import SCRFD, Threshold

class SCRFDDetector:
    def __init__(self, confidence_threshold=0.5, input_size=(512, 512)):
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.threshold = Threshold(probability=confidence_threshold)
        self.model_loaded = False
        self.model = None
    
    def load_model(self):
        self.model = SCRFD.from_path("./models/scrfd_models/scrfd.onnx")
        self.model_loaded = True
    
    def detect_faces(self, image_path):
        if not self.model_loaded:
            self.load_model()
        
        image = Image.open(image_path).convert("RGB")   
            
        faces = self.model.detect(image, threshold=self.threshold)
        detections = []

        for face in faces:
            bbox = face.bbox
            x1, y1 = bbox.upper_left.x, bbox.upper_left.y
            x2, y2 = bbox.lower_right.x, bbox.lower_right.y

            w, h = (x2 - x1), (y2 - y1)

            confidence = face.probability
        
            detections.append({
                'bbox': (x1, y1, w, h),
                'confidence': float(confidence)
            })

        return detections
