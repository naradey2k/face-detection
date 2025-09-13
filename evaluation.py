import numpy as np
import time
from sklearn.metrics import precision_score, recall_score
import os

class FaceDetectionEvaluator:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        
    def calculate_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections(self, ground_truth, detections):
        matched_gt = [False] * len(ground_truth)
        matched_detections = [False] * len(detections)
        
        sorted_detections = sorted(enumerate(detections), 
                                 key=lambda x: x[1].get('confidence', 0), 
                                 reverse=True)
        
        for det_idx, detection in sorted_detections:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(ground_truth):
                if matched_gt[gt_idx]:
                    continue
                    
                iou = self.calculate_iou(gt_box, detection['bbox'])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx != -1:
                matched_gt[best_gt_idx] = True
                matched_detections[det_idx] = True
                
        return matched_gt, matched_detections
    
    def evaluate_model(self, model, data_path, annotations):
        total_gt_faces = 0
        total_detections = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        processing_times = []
        
        print(f"Evaluating {len(annotations)} images...")
        
        for i, annotation in enumerate(annotations): 
            img_path = os.path.join(data_path, annotation['img_path'].replace('\\', '/'))
            
            gt_bboxes = annotation['annotations']['bbox']
            gt_invalid = annotation['annotations']['invalid']
            
            valid_gt_bboxes = [bbox for bbox, invalid in zip(gt_bboxes, gt_invalid) if not invalid]
            total_gt_faces += len(valid_gt_bboxes)
            
            start_time = time.time()
            
            detections = model.detect_faces(img_path)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            total_detections += len(detections)
            
            matched_gt, matched_detections = self.match_detections(valid_gt_bboxes, detections)
            
            true_positives += sum(matched_detections)
            false_positives += len(detections) - sum(matched_detections)
            false_negatives += len(valid_gt_bboxes) - sum(matched_gt)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        total_processing_time = sum(processing_times)
        
        return {
            'precision': precision,
            'recall': recall,
            'total_gt_faces': total_gt_faces,
            'total_detections': total_detections,
            'avg_processing_time_per_image': avg_processing_time,
            'total_processing_time': total_processing_time,
            'images_processed': len(processing_times)
        }
    
    def evaluate_all_models(self, models, data_path, annotations):
        results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            results[model_name] = self.evaluate_model(model, data_path, annotations)
            
        return results
