from mtcnn import MTCNN
from mtcnn.utils.images import load_image

class MTCNNDetector():
    def __init__(self, confidence_threshold=0.5, input_size=(512, 512)):
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.model_loaded = False
        self.model = None
        
    def load_model(self):
        self.model = MTCNN()
        self.model_loaded = True
                
    def detect_faces(self, image_path):
        if not self.model_loaded:
            self.load_model()

        image = load_image(image_path)  
        
        detections = self.model.detect_faces(image)
        faces = []

        for face_data in detections:
            x1, y1, x2, y2 = face_data['box']
            confidence = face_data['confidence']
            
            if confidence >= self.confidence_threshold:
                # w, h = int(x2 - x1), int(y2 - y1)
                faces.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(confidence),
                })

        return faces
        