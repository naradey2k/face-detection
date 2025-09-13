
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from models.opencv_haar import OpenCVHaarDetector
from models.retina_face import RetinaFaceDetector
from models.opencv_dnn import OpenCVDNNDetector
from models.scrfd import SCRFDDetector
from models.mtcnn import MTCNNDetector
# from models.blazeface import BlazeFaceDetector

opencv_haar_model = OpenCVHaarDetector()
opencv_haar_model.load_model()

opencv_dnn_model = OpenCVDNNDetector()
opencv_dnn_model.load_model()

retina_face_model = RetinaFaceDetector()
retina_face_model.load_model()

scrfd_model = SCRFDDetector()
scrfd_model.load_model()

mt = MTCNNDetector()
mt.load_model()


with open("data/annos.json", "r") as f:
    data = json.load(f)

    for instance in data[2:]:
        path = os.path.join("data", instance["img_path"].replace("\\", "/"))
        bboxes = instance["annotations"]["bbox"]
        blur = instance["annotations"]["blur"]
        pose = instance["annotations"]["pose"]
        invalid = instance["annotations"]["invalid"] 

        img = Image.open(path)

        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(img)

        opencv_haar_annotations = opencv_haar_model.detect_faces(path)
        opencv_dnn_annotations = opencv_dnn_model.detect_faces(path)
        # bb = blaze.detect_faces(path)
        # print(bb)
        retina_face_annotations = retina_face_model.detect_faces(path)
        scrfd_annotations = scrfd_model.detect_faces(path)
        mts = mt.detect_faces(path)
        # print(opencv_haar_annotations)
        # print(opencv_dnn_annotations)
        # print(retina_face_annotations)
        # print(type(retina_face_annotations))

        #mtcnn
        for i, detection in enumerate(mts):
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            # color = "red" if invalid[i] else "blue"
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="blue", facecolor="none")
            ax.add_patch(rect)
            
            label = f"Confidence: {confidence:.2f}"
            ax.text(x, y-5, label, color="blue", fontsize=8, backgroundcolor="white")

        # SCRFD
        # for i, detection in enumerate(scrfd_annotations):
        #     x, y, w, h = detection['bbox']
        #     confidence = detection['confidence']
        #     # color = "red" if invalid[i] else "blue"
        #     rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="blue", facecolor="none")
        #     ax.add_patch(rect)
            
        #     label = f"Confidence: {confidence:.2f}"
        #     ax.text(x, y-5, label, color="blue", fontsize=8, backgroundcolor="white")    

        # RetinaFace
        # for i, detection in enumerate(retina_face_annotations):
        #     x, y, w, h = detection['bbox']
        #     confidence = detection['confidence']
        #     color = "red" if invalid[i] else "blue"
        #     rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
        #     ax.add_patch(rect)
            
        #     label = f"Confidence: {confidence:.2f}"
        #     ax.text(x, y-5, label, color=color, fontsize=8, backgroundcolor="white")    

        # DNN
        # for i, detection in enumerate(opencv_dnn_annotations):
        #     x, y, w, h = detection['bbox']
        #     confidence = detection['confidence']
        #     color = "red" if invalid[i] else "blue"
        #     rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
        #     ax.add_patch(rect)
            
        #     label = f"Confidence: {confidence:.2f}"
        #     ax.text(x, y-5, label, color=color, fontsize=8, backgroundcolor="white")

        # Haar
        # for i, (x, y, w, h) in enumerate(bboxes):
        #     color = "red" if invalid[i] else "blue"
        #     rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
        #     ax.add_patch(rect)
            
        #     label = f"Pose:{pose[i]} Blur:{blur[i]}"
        #     ax.text(x, y-5, label, color=color, fontsize=8, backgroundcolor="white")

        plt.show()
        # break
