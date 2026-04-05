from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from .ambulance_classifier import AmbulanceClassifier
from .light_detector import EmergencyLightDetector


@dataclass
class DetectionResult:
    detected: bool
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]]
    road_direction: str


class VehicleDetector:
    """
    YOLOv8-based vehicle detector focused on ambulances with basic false-positive filtering.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        bbox_min_area: int = 8000,
        aspect_ratio_range: Tuple[float, float] = (0.5, 4.0),
        custom_model_path: Optional[str] = None,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.bbox_min_area = int(bbox_min_area)
        self.aspect_ratio_range = aspect_ratio_range
        self.model = None
        self.predict_lock = threading.Lock()
        self.light_detector = EmergencyLightDetector()
        self.enable_light_detector = False  # Set to False to "put on hold" as requested

        if YOLO is not None:
            model_path = None
            if custom_model_path and os.path.exists(custom_model_path):
                model_path = custom_model_path
            else:
                # Look in the same directory as this file's parent's models folder
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                default_custom = os.path.join(base_dir, "models", "ambulance_best.pt")
                if os.path.exists(default_custom):
                    model_path = default_custom
            try:
                print(f"Loading YOLO model from: {model_path if model_path else 'yolov8n.pt'}")
                self.model = YOLO(model_path if model_path else "yolov8n.pt")
                print(f"Model task: {getattr(self.model, 'task', 'unknown')}")
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                self.model = None

        self.classifier = AmbulanceClassifier(self.model.names if getattr(self.model, "names", None) else {})

    def _filter_bbox(self, bbox: Tuple[int, int, int, int]) -> bool:
        x, y, w, h = bbox
        area = w * h
        if area < self.bbox_min_area:
            return False
        ratio = w / float(h) if h > 0 else 0.0
        if ratio < self.aspect_ratio_range[0] or ratio > self.aspect_ratio_range[1]:
            return False
        return True

    def detect(self, frame: np.ndarray, road_direction: str) -> DetectionResult:
        """
        Runs inference on a frame and returns a filtered ambulance detection result.
        Combines YOLO vehicle detection with flashing light detection.
        """
        # 1. Run Flashing Light Detection
        if self.enable_light_detector:
            light_detected, light_conf = self.light_detector.detect_flashing(frame, road_direction)
        else:
            light_detected, light_conf = False, 0.0

        # 2. Run YOLO Vehicle Detection
        if self.model is None:
            if light_detected:
                return DetectionResult(True, light_conf, None, road_direction)
            return DetectionResult(False, 0.0, None, road_direction)

        try:
            with self.predict_lock:
                # Use a slightly larger imgsz or the default for better feature extraction
                # but consider the classification model was trained at 224
                imgsz = 224 if getattr(self.model, "task", "") == "classify" else 640
                results = self.model.predict(frame, imgsz=imgsz, conf=self.confidence_threshold, verbose=False)
        except Exception:
            if light_detected:
                return DetectionResult(True, light_conf, None, road_direction)
            return DetectionResult(False, 0.0, None, road_direction)

        best_conf = 0.0
        best_bbox = None
        if results:
            for r in results:
                # Handle Classification Task
                if getattr(r, "probs", None) is not None:
                    probs = r.probs
                    top1_idx = int(probs.top1)
                    top1_conf = float(probs.top1conf)
                    label = str(self.model.names.get(top1_idx, str(top1_idx))).lower()
                    
                    if self.classifier.is_ambulance_label(label) and top1_conf > self.confidence_threshold:
                        print(f"[{road_direction}] Detected {label} with confidence {top1_conf:.2f}")
                        best_conf = top1_conf
                        # Classification doesn't provide a bounding box
                        best_bbox = (0, 0, 0, 0) 
                    elif self.classifier.is_ambulance_label(label):
                        print(f"[{road_direction}] Classified as {label} but confidence {top1_conf:.2f} < {self.confidence_threshold}")
                    continue

                # Handle Detection Task
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                xywh = boxes.xywh.cpu().numpy() if hasattr(boxes, "xywh") else None
                confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else None
                clses = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else None
                if xywh is None or confs is None or clses is None:
                    continue
                for i in range(len(xywh)):
                    cx, cy, w, h = xywh[i]
                    conf = float(confs[i])
                    cls_id = int(clses[i])
                    label = str(self.model.names.get(cls_id, str(cls_id))).lower() if getattr(self.model, "names", None) else str(cls_id)
                    if not self.classifier.is_ambulance_label(label):
                        continue
                    x = int(max(0, cx - w / 2))
                    y = int(max(0, cy - h / 2))
                    bbox = (x, y, int(w), int(h))
                    if not self._filter_bbox(bbox):
                        continue
                    if conf > best_conf:
                        best_conf = conf
                        best_bbox = bbox

        # Combine results
        is_detected = best_bbox is not None or light_detected
        
        # If both are detected, boost confidence
        if best_bbox is not None and light_detected:
            final_conf = min(1.0, best_conf + (light_conf * 0.2))
        elif best_bbox is not None:
            final_conf = best_conf
        else:
            final_conf = light_conf

        return DetectionResult(is_detected, final_conf, best_bbox if best_bbox != (0, 0, 0, 0) else None, road_direction)
