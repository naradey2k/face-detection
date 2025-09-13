import cv2

class OpenCVHaarDetector():
    def __init__(self, confidence_threshold=0.5, 
                 scale_factor=1.1, 
                 min_neighbors=3,
                 min_size=(30, 30)):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.cascade = None
        self.model_loaded = False
        
    def load_model(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.model_loaded = True

    def detect_faces(self, image_path):
        if not self.model_loaded:
            self.load_model()
        
        image = cv2.imread(image_path)
        if image is None:
            return []
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        detections = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to standard format
        formatted_detections = []
        for (x, y, w, h) in detections:
            formatted_detections.append({
                'bbox': (x, y, w, h),
                'confidence': 1.0  # Haar cascades don't provide confidence scores
            })
        
        return formatted_detections