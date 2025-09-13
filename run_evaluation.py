import json
import time
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

from models.opencv_haar import OpenCVHaarDetector
from models.opencv_dnn import OpenCVDNNDetector
from models.retina_face import RetinaFaceDetector
from models.scrfd import SCRFDDetector
from models.mtcnn import MTCNNDetector

from evaluation import FaceDetectionEvaluator

def load_annotations():
    with open("data/annos.json", 'r') as f:
        return json.load(f)

def initialize_models():
    models = {}
    
    print("Loading OpenCV Haar Cascade")
    haar_detector = OpenCVHaarDetector()
    haar_detector.load_model()
    models['OpenCV Haar'] = haar_detector
    print("OpenCV Haar loaded successfully")
    
    print("Loading OpenCV DNN")
    dnn_detector = OpenCVDNNDetector()
    dnn_detector.load_model()
    models['OpenCV DNN'] = dnn_detector
    print("OpenCV DNN loaded successfully")
    
    print("Loading RetinaFace")
    retina_detector = RetinaFaceDetector()
    retina_detector.load_model()
    models['RetinaFace'] = retina_detector
    print("RetinaFace loaded successfully")
    
    print("Loading SCRFD")
    scrfd_detector = SCRFDDetector()
    scrfd_detector.load_model()
    models['SCRFD'] = scrfd_detector
    print("SCRFD loaded successfully")

    print("Loading MTCNN")
    mtcnn_detector = MTCNNDetector()
    mtcnn_detector.load_model()
    models['MTCNN'] = mtcnn_detector
    print("MTCNN loaded successfully")
    
    return models

def run_evaluation(models, data_path, annotations, iou_threshold=0.5):
    print(f"\nStarting evaluation with IoU threshold: {iou_threshold}")
    print(f"Processing all {len(annotations)} images")
    
    evaluator = FaceDetectionEvaluator(iou_threshold=iou_threshold)
    
    start_time = time.time()
    results = evaluator.evaluate_all_models(models, data_path, annotations)
    total_time = time.time() - start_time
    
    print(f"\nEvaluation completed in {total_time:.2f} seconds")
    return results

def print_results(results):
    print("\n" + "="*100)
    print("EVALUATION RESULTS")
    print("="*100)
    
    print(f"{'Model':<15} {'Precision':<10} {'Recall':<10} {'Avg Time':<10} {'Total Time':<10}")
    print("-"*100)
    
    for model_name, result in results.items():
        print(f"{model_name:<15} "
              f"{result['precision']:<10.4f} "
              f"{result['recall']:<10.4f} "
              f"{result['avg_processing_time_per_image']:<10.4f} "
              f"{result['total_processing_time']:<10}"
              )
    
    print("="*100)

def visualize_random_image(annotations, models):
    valid_annotations = []
    for annotation in annotations:
        gt_bboxes = annotation['annotations']['bbox']
        gt_invalid = annotation['annotations']['invalid']
        valid_gt_bboxes = [bbox for bbox, invalid in zip(gt_bboxes, gt_invalid) if not invalid]
        
        if len(valid_gt_bboxes) > 0:  
            valid_annotations.append(annotation)
    
    if not valid_annotations:
        print("No valid annotations found!")
        return
    
    random_annotation = random.choice(valid_annotations)
    
    img_path = os.path.join("data", random_annotation['img_path'].replace('\\', '/'))
    
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    print(f"\nSelected random image: {os.path.basename(img_path)}")
    
    gt_bboxes = random_annotation['annotations']['bbox']
    gt_invalid = random_annotation['annotations']['invalid']
    valid_gt_bboxes = [bbox for bbox, invalid in zip(gt_bboxes, gt_invalid) if not invalid]
    
    print(f"Ground truth faces: {len(valid_gt_bboxes)}")
    
    img = Image.open(img_path)
    
    num_models = len(models)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    ax = axes[0]
    ax.imshow(img)
    ax.set_title("Ground Truth", fontsize=14, fontweight='bold', color='green')
    ax.axis('off')
    
    for i, bbox in enumerate(valid_gt_bboxes):
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=3, 
                               edgecolor='green', facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y-5, f"GT {i+1}", color='green', fontsize=10, 
               fontweight='bold', backgroundcolor='white', alpha=0.8)
    
    colors = ['red', 'blue', 'orange', 'purple', 'brown', 'black', 'pinks']
    model_names = list(models.keys())
    
    for i, (model_name, model) in enumerate(models.items()):
        ax = axes[i + 1]
        ax.imshow(img)
        ax.set_title(f"{model_name}", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        color = "black"
        
        model_detections = model.detect_faces(img_path)
        print(f"{model_name}: {len(model_detections)} faces detected")
        
        for j, detection in enumerate(model_detections):
            x, y, w, h = detection['bbox']
            confidence = detection.get('confidence', 0.0)
            
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                    edgecolor=color, facecolor=color, alpha=0.3)
            ax.add_patch(rect)
            
            label = f"{j+1}\n{confidence:.2f}"
            ax.text(x, y-5, label, color=color, fontsize=9, 
                    fontweight='bold', backgroundcolor='white', alpha=0.8)
        
    for si in range(len(models) + 1, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Face Detection Comparison - {os.path.basename(img_path)}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main():
    print("\nLoading annotations...")
    annotations = load_annotations()
    print(f"Loaded {len(annotations)} image annotations")
    
    models = initialize_models()
    
    print("\n" + "="*50)
    print("VISUALIZING RANDOM IMAGE")
    print("="*50)
    visualize_random_image(annotations, models)
    
    print("\n" + "="*50)
    print("RUNNING EVALUATION")
    print("="*50)
    results = run_evaluation(models, data_path="data", annotations=annotations)
    
    print_results(results)
    
    print("\nEvaluation completed successfully!")
    return 0

if __name__ == "__main__":
    main()
