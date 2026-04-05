import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional

class EmergencyLightDetector:
    """
    Detects flashing red and blue emergency lights in a video stream.
    Uses color thresholding and frequency analysis.
    """
    def __init__(self, history_size: int = 15, threshold: float = 0.4):
        self.history_size = history_size
        self.threshold = threshold
        # History of color detection (True/False) per road
        self.red_history: Dict[str, List[bool]] = {}
        self.blue_history: Dict[str, List[bool]] = {}
        
        # HSV color ranges for Red and Blue
        # Red has two ranges in HSV
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.blue_lower = np.array([100, 150, 150])
        self.blue_upper = np.array([140, 255, 255])

    def _detect_color(self, hsv_frame: np.ndarray, lower: np.ndarray, upper: np.ndarray, min_area: int = 50) -> bool:
        mask = cv2.inRange(hsv_frame, lower, upper)
        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                return True
        return False

    def detect_flashing(self, frame: np.ndarray, road_id: str) -> Tuple[bool, float]:
        """
        Detects if flashing red/blue lights are present on a specific road.
        Returns (detected, confidence)
        """
        if road_id not in self.red_history:
            self.red_history[road_id] = []
            self.blue_history[road_id] = []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect current presence
        red_present = self._detect_color(hsv, self.red_lower1, self.red_upper1) or \
                      self._detect_color(hsv, self.red_lower2, self.red_upper2)
        blue_present = self._detect_color(hsv, self.blue_lower, self.blue_upper)

        # Update history
        self.red_history[road_id].append(red_present)
        self.blue_history[road_id].append(blue_present)
        
        if len(self.red_history[road_id]) > self.history_size:
            self.red_history[road_id].pop(0)
            self.blue_history[road_id].pop(0)

        # Analysis: Flashing means toggling state
        # A simple way to detect flashing is to check if the state has changed 
        # multiple times in the history window.
        def count_transitions(history: List[bool]) -> int:
            return sum(1 for i in range(1, len(history)) if history[i] != history[i-1])

        red_transitions = count_transitions(self.red_history[road_id])
        blue_transitions = count_transitions(self.blue_history[road_id])
        
        # If we have both colors and they are changing state, or one is very active
        is_flashing = (red_transitions > 2 and blue_transitions > 2) or \
                      (red_transitions > 4) or (blue_transitions > 4)
        
        confidence = min(1.0, (red_transitions + blue_transitions) / 10.0)
        
        return is_flashing, confidence
