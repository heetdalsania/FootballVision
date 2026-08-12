"""
Evaluation Metrics Module for NFL Vision App
Computes detection, tracking, and classification metrics.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class DetectionMetrics:
    """Detection evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    mAP: float  # Mean Average Precision
    mAP_50: float  # mAP at IoU 0.5
    mAP_75: float  # mAP at IoU 0.75


@dataclass
class TrackingMetrics:
    """Tracking evaluation metrics (MOT metrics)."""
    mota: float  # Multi-Object Tracking Accuracy
    motp: float  # Multi-Object Tracking Precision
    idf1: float  # ID F1 Score
    id_switches: int
    mostly_tracked: int
    mostly_lost: int
    fragmentations: int


@dataclass
class ClassificationMetrics:
    """Formation classification metrics."""
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: np.ndarray
    per_class_accuracy: Dict[str, float]


class Evaluator:
    """
    Evaluation harness for NFL Vision models.
    
    Computes metrics for:
    - Detection: mAP, precision, recall
    - Tracking: MOTA, MOTP, IDF1
    - Classification: Formation accuracy
    
    Example:
        evaluator = Evaluator()
        det_metrics = evaluator.compute_mAP(predictions, ground_truth)
        track_metrics = evaluator.compute_mota(tracks, gt_tracks)
        cls_metrics = evaluator.compute_formation_accuracy(preds, labels)
    """
    
    def __init__(self, iou_thresholds: Optional[List[float]] = None):
        """
        Initialize evaluator.
        
        Args:
            iou_thresholds: IoU thresholds for mAP calculation.
        """
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    # ==================== DETECTION METRICS ====================
    
    def compute_iou(self, 
                    box1: Tuple[int, int, int, int],
                    box2: Tuple[int, int, int, int]) -> float:
        """
        Compute Intersection over Union between two boxes.
        
        Args:
            box1: (x1, y1, x2, y2) format
            box2: (x1, y1, x2, y2) format
        
        Returns:
            IoU value [0, 1]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-6)
    
    def compute_ap(self,
                   predictions: List[Dict],
                   ground_truth: List[Dict],
                   iou_threshold: float = 0.5) -> float:
        """
        Compute Average Precision at a single IoU threshold.
        
        Args:
            predictions: List of {"bbox": (x1,y1,x2,y2), "confidence": float}
            ground_truth: List of {"bbox": (x1,y1,x2,y2)}
            iou_threshold: IoU threshold for matching
        
        Returns:
            AP value [0, 1]
        """
        if not predictions or not ground_truth:
            return 0.0
        
        # Sort predictions by confidence
        predictions = sorted(predictions, key=lambda x: -x["confidence"])
        
        # Track matched ground truth
        gt_matched = [False] * len(ground_truth)
        
        tp = []
        fp = []
        
        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_matched[gt_idx]:
                    continue
                
                iou = self.compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp.append(1)
                fp.append(0)
                gt_matched[best_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        
        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(ground_truth)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP using all-points interpolation
        recalls = np.concatenate([[0], recalls, [1]])
        precisions = np.concatenate([[0], precisions, [0]])
        
        # Ensure precision is monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        # Find points where recall changes
        recall_change = np.where(recalls[1:] != recalls[:-1])[0]
        
        # Compute area under curve
        ap = np.sum((recalls[recall_change + 1] - recalls[recall_change]) * precisions[recall_change + 1])
        
        return ap
    
    def compute_mAP(self,
                    all_predictions: List[List[Dict]],
                    all_ground_truth: List[List[Dict]]) -> DetectionMetrics:
        """
        Compute mean Average Precision across multiple frames.
        
        Args:
            all_predictions: List of frame predictions
            all_ground_truth: List of frame ground truth
        
        Returns:
            DetectionMetrics with mAP values
        """
        # Flatten all predictions and ground truth per class
        # For simplicity, assuming single class here
        
        aps = {thresh: [] for thresh in self.iou_thresholds}
        
        for preds, gts in zip(all_predictions, all_ground_truth):
            for thresh in self.iou_thresholds:
                ap = self.compute_ap(preds, gts, thresh)
                aps[thresh].append(ap)
        
        # Compute mean AP per threshold
        mean_aps = {thresh: np.mean(ap_list) if ap_list else 0.0 
                    for thresh, ap_list in aps.items()}
        
        # Overall mAP is mean across thresholds
        mAP = np.mean(list(mean_aps.values()))
        mAP_50 = mean_aps.get(0.5, 0.0)
        mAP_75 = mean_aps.get(0.75, 0.0)
        
        # Compute precision/recall at IoU 0.5
        all_tp = 0
        all_fp = 0
        all_fn = 0
        
        for preds, gts in zip(all_predictions, all_ground_truth):
            preds_sorted = sorted(preds, key=lambda x: -x["confidence"])
            gt_matched = [False] * len(gts)
            
            for pred in preds_sorted:
                matched = False
                for gt_idx, gt in enumerate(gts):
                    if gt_matched[gt_idx]:
                        continue
                    if self.compute_iou(pred["bbox"], gt["bbox"]) >= 0.5:
                        gt_matched[gt_idx] = True
                        matched = True
                        break
                
                if matched:
                    all_tp += 1
                else:
                    all_fp += 1
            
            all_fn += sum(1 for m in gt_matched if not m)
        
        precision = all_tp / max(all_tp + all_fp, 1)
        recall = all_tp / max(all_tp + all_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            mAP=mAP,
            mAP_50=mAP_50,
            mAP_75=mAP_75
        )
    
    # ==================== TRACKING METRICS ====================
    
    def compute_mota(self,
                     predicted_tracks: List[Dict[int, Tuple]],
                     ground_truth_tracks: List[Dict[int, Tuple]],
                     iou_threshold: float = 0.5) -> TrackingMetrics:
        """
        Compute Multi-Object Tracking Accuracy (MOTA).
        
        MOTA = 1 - (FN + FP + ID_SW) / GT_TOTAL
        
        Args:
            predicted_tracks: List of frame dicts {track_id: bbox}
            ground_truth_tracks: List of frame dicts {gt_id: bbox}
            iou_threshold: IoU threshold for matching
        
        Returns:
            TrackingMetrics with MOTA, MOTP, IDF1
        """
        total_gt = 0
        false_negatives = 0
        false_positives = 0
        id_switches = 0
        total_iou = 0
        matches = 0
        
        # Track ID associations
        prev_assignments: Dict[int, int] = {}  # gt_id -> pred_id
        
        gt_track_lengths: Dict[int, int] = defaultdict(int)
        gt_track_matched: Dict[int, int] = defaultdict(int)
        
        for frame_idx, (pred_frame, gt_frame) in enumerate(zip(predicted_tracks, ground_truth_tracks)):
            gt_ids = list(gt_frame.keys())
            pred_ids = list(pred_frame.keys())
            
            total_gt += len(gt_ids)
            
            # Track GT track lengths
            for gt_id in gt_ids:
                gt_track_lengths[gt_id] += 1
            
            # Compute IoU matrix
            iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))
            for i, gt_id in enumerate(gt_ids):
                for j, pred_id in enumerate(pred_ids):
                    iou_matrix[i, j] = self.compute_iou(gt_frame[gt_id], pred_frame[pred_id])
            
            # Greedy matching
            matched_gt = set()
            matched_pred = set()
            curr_assignments = {}
            
            # Sort by IoU
            indices = np.argsort(-iou_matrix.flatten())
            for idx in indices:
                i, j = divmod(idx, len(pred_ids)) if pred_ids else (0, 0)
                if not pred_ids:
                    break
                
                if i < len(gt_ids) and j < len(pred_ids):
                    if i not in matched_gt and j not in matched_pred:
                        if iou_matrix[i, j] >= iou_threshold:
                            gt_id = gt_ids[i]
                            pred_id = pred_ids[j]
                            
                            matched_gt.add(i)
                            matched_pred.add(j)
                            curr_assignments[gt_id] = pred_id
                            
                            total_iou += iou_matrix[i, j]
                            matches += 1
                            
                            gt_track_matched[gt_id] += 1
                            
                            # Check for ID switch
                            if gt_id in prev_assignments:
                                if prev_assignments[gt_id] != pred_id:
                                    id_switches += 1
            
            # Count FN and FP
            false_negatives += len(gt_ids) - len(matched_gt)
            false_positives += len(pred_ids) - len(matched_pred)
            
            prev_assignments = curr_assignments
        
        # Compute metrics
        motp = total_iou / max(matches, 1)
        mota = 1 - (false_negatives + false_positives + id_switches) / max(total_gt, 1)
        
        # Compute IDF1
        # IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
        idtp = sum(gt_track_matched.values())
        idfn = sum(gt_track_lengths[gid] - gt_track_matched[gid] for gid in gt_track_lengths)
        idfp = false_positives
        idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)
        
        # Mostly tracked / mostly lost
        mostly_tracked = sum(1 for gid in gt_track_lengths 
                            if gt_track_matched[gid] / gt_track_lengths[gid] >= 0.8)
        mostly_lost = sum(1 for gid in gt_track_lengths 
                         if gt_track_matched[gid] / gt_track_lengths[gid] <= 0.2)
        
        return TrackingMetrics(
            mota=mota,
            motp=motp,
            idf1=idf1,
            id_switches=id_switches,
            mostly_tracked=mostly_tracked,
            mostly_lost=mostly_lost,
            fragmentations=0  # Simplified
        )
    
    def compute_idf1(self,
                     predicted_tracks: List[Dict[int, Tuple]],
                     ground_truth_tracks: List[Dict[int, Tuple]]) -> float:
        """Compute IDF1 score."""
        metrics = self.compute_mota(predicted_tracks, ground_truth_tracks)
        return metrics.idf1
    
    # ==================== CLASSIFICATION METRICS ====================
    
    def compute_formation_accuracy(self,
                                   predictions: List[int],
                                   ground_truth: List[int],
                                   class_names: Optional[List[str]] = None) -> ClassificationMetrics:
        """
        Compute formation classification metrics.
        
        Args:
            predictions: List of predicted class indices
            ground_truth: List of ground truth class indices
            class_names: Optional list of class names
        
        Returns:
            ClassificationMetrics with accuracy, F1, confusion matrix
        """
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        num_classes = max(predictions.max(), ground_truth.max()) + 1
        
        # Confusion matrix
        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for pred, gt in zip(predictions, ground_truth):
            confusion[gt, pred] += 1
        
        # Overall accuracy
        accuracy = np.sum(predictions == ground_truth) / len(predictions)
        
        # Per-class metrics
        per_class_precision = []
        per_class_recall = []
        per_class_accuracy = {}
        
        for c in range(num_classes):
            tp = confusion[c, c]
            fp = confusion[:, c].sum() - tp
            fn = confusion[c, :].sum() - tp
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            
            per_class_precision.append(precision)
            per_class_recall.append(recall)
            
            class_name = class_names[c] if class_names and c < len(class_names) else str(c)
            per_class_accuracy[class_name] = tp / max(confusion[c, :].sum(), 1)
        
        # Macro averages
        precision_macro = np.mean(per_class_precision)
        recall_macro = np.mean(per_class_recall)
        f1_macro = 2 * precision_macro * recall_macro / max(precision_macro + recall_macro, 1e-6)
        
        return ClassificationMetrics(
            accuracy=accuracy,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            f1_macro=f1_macro,
            confusion_matrix=confusion,
            per_class_accuracy=per_class_accuracy
        )
    
    # ==================== REPORTING ====================
    
    def generate_report(self,
                        detection_metrics: Optional[DetectionMetrics] = None,
                        tracking_metrics: Optional[TrackingMetrics] = None,
                        classification_metrics: Optional[ClassificationMetrics] = None) -> str:
        """
        Generate a formatted evaluation report.
        
        Returns:
            Formatted string report.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("NFL VISION - EVALUATION REPORT")
        lines.append("=" * 60)
        
        if detection_metrics:
            lines.append("\n📊 DETECTION METRICS")
            lines.append("-" * 40)
            lines.append(f"  mAP:          {detection_metrics.mAP:.4f}")
            lines.append(f"  mAP@50:       {detection_metrics.mAP_50:.4f}")
            lines.append(f"  mAP@75:       {detection_metrics.mAP_75:.4f}")
            lines.append(f"  Precision:    {detection_metrics.precision:.4f}")
            lines.append(f"  Recall:       {detection_metrics.recall:.4f}")
            lines.append(f"  F1 Score:     {detection_metrics.f1_score:.4f}")
        
        if tracking_metrics:
            lines.append("\n🎯 TRACKING METRICS")
            lines.append("-" * 40)
            lines.append(f"  MOTA:         {tracking_metrics.mota:.4f}")
            lines.append(f"  MOTP:         {tracking_metrics.motp:.4f}")
            lines.append(f"  IDF1:         {tracking_metrics.idf1:.4f}")
            lines.append(f"  ID Switches:  {tracking_metrics.id_switches}")
            lines.append(f"  Mostly Track: {tracking_metrics.mostly_tracked}")
            lines.append(f"  Mostly Lost:  {tracking_metrics.mostly_lost}")
        
        if classification_metrics:
            lines.append("\n🏈 FORMATION CLASSIFICATION METRICS")
            lines.append("-" * 40)
            lines.append(f"  Accuracy:     {classification_metrics.accuracy:.4f}")
            lines.append(f"  Precision:    {classification_metrics.precision_macro:.4f}")
            lines.append(f"  Recall:       {classification_metrics.recall_macro:.4f}")
            lines.append(f"  F1 Score:     {classification_metrics.f1_macro:.4f}")
            
            lines.append("\n  Per-class accuracy:")
            for cls_name, acc in classification_metrics.per_class_accuracy.items():
                lines.append(f"    {cls_name}: {acc:.4f}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage
    print("NFL Vision - Evaluator Demo")
    print("-" * 40)
    
    evaluator = Evaluator()
    
    # Test detection metrics
    predictions = [
        [{"bbox": (100, 100, 150, 200), "confidence": 0.9},
         {"bbox": (300, 150, 350, 250), "confidence": 0.8}]
    ]
    ground_truth = [
        [{"bbox": (102, 98, 148, 202)},
         {"bbox": (305, 155, 355, 255)}]
    ]
    
    det_metrics = evaluator.compute_mAP(predictions, ground_truth)
    print(f"\nDetection mAP@50: {det_metrics.mAP_50:.4f}")
    
    # Test classification metrics
    preds = [0, 1, 1, 2, 0, 1, 2, 2, 0, 1]
    labels = [0, 1, 0, 2, 0, 1, 2, 1, 0, 1]
    
    cls_metrics = evaluator.compute_formation_accuracy(
        preds, labels, 
        class_names=["shotgun", "i_formation", "pro_set"]
    )
    print(f"Classification Accuracy: {cls_metrics.accuracy:.4f}")
    
    # Generate report
    print("\n" + evaluator.generate_report(
        detection_metrics=det_metrics,
        classification_metrics=cls_metrics
    ))
