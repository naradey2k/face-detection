import cv2
import numpy as np
import os
from batch_face import RetinaFace

class RetinaFaceDetector():
    def __init__(self, confidence_threshold=0.5, input_size=(512, 512)):
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.model_loaded = False
        self.model = None
        
    def load_model(self):
        self.model = RetinaFace(gpu_id=-1, network="resnet50")
        self.model_loaded = True
                
    def detect_faces(self, image_path):
        if not self.model_loaded:
            self.load_model()

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
        
        max_size = max(self.input_size)
        detections = self.model(image, threshold=self.confidence_threshold, max_size=max_size, return_dict=True)
        
        faces = []
        for face_data in detections:
            x1, y1, x2, y2 = face_data['box']
            w = x2 - x1
            h = y2 - y1
            confidence = face_data['score']
            
            if confidence >= self.confidence_threshold:
                faces.append({
                    'bbox': (int(x1), int(y1), int(w), int(h)),
                    'confidence': float(confidence),
                })

        return faces
        