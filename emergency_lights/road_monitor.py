import threading
import time
from typing import Dict

from .light_controller import LightController


class RoadMonitor:
    """
    Consumes detection events and updates the light controller. Prints state transitions.
    """

    def __init__(self, controller: LightController) -> None:
        self.controller = controller
        self._lock = threading.Lock()
        self._last_active: Dict[str, bool] = {}

    def update(self, road: str, detected: bool, confidence: float) -> None:
        with self._lock:
            prev = self._last_active.get(road, False)
            if detected:
                self.controller.turn_on(road)
            else:
                self.controller.turn_off(road)
            curr = detected
            if curr != prev:
                ts = time.strftime("%H:%M:%S", time.localtime())
                state = "ON" if curr else "OFF"
                print(f"[{ts}] {road}: EMERGENCY LIGHT {state} (confidence={confidence:.2f})")
            self._last_active[road] = curr
