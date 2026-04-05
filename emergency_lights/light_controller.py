import threading
import time
from typing import Dict, List, Tuple


class LightController:
    """
    Thread-safe emergency light state manager for four roads.
    """

    def __init__(self, roads: List[str]) -> None:
        self._lock = threading.Lock()
        self._roads = list(roads)
        self._active: Dict[str, bool] = {r: False for r in self._roads}
        self._hold_until: Dict[str, float] = {r: 0.0 for r in self._roads}
        self._last_change: Dict[str, float] = {r: 0.0 for r in self._roads}
        self._flash_period = 0.3
        self._hold_duration = 30.0

    def turn_on(self, road: str) -> None:
        with self._lock:
            if road in self._active:
                self._active[road] = True
                self._hold_until[road] = time.time() + self._hold_duration
                self._last_change[road] = time.time()

    def turn_off(self, road: str) -> None:
        with self._lock:
            if road in self._active:
                self._active[road] = False
                # We don't clear _hold_until here so it stays on for the remainder of the 30s
                self._last_change[road] = time.time()

    def reset(self) -> None:
        with self._lock:
            for r in self._roads:
                self._active[r] = False
                self._hold_until[r] = 0.0
                self._last_change[r] = time.time()

    def is_on(self, road: str) -> bool:
        with self._lock:
            return time.time() < self._hold_until.get(road, 0.0)

    def get_active_roads(self) -> List[str]:
        with self._lock:
            now = time.time()
            return [r for r in self._roads if now < self._hold_until.get(r, 0.0)]

    def get_visual_state(self, road: str) -> Tuple[str, Tuple[int, int, int]]:
        """
        Returns a tuple of state and RGB color for rendering.
        """
        with self._lock:
            if road not in self._roads:
                return "OFF", (128, 128, 128)
            
            now = time.time()
            if now >= self._hold_until.get(road, 0.0):
                return "OFF", (128, 128, 128)
                
            phase = int((now / self._flash_period)) % 2
            return ("ON", (0, 102, 255)) if phase == 0 else ("ON", (255, 45, 45))

    def get_all_states(self) -> Dict[str, str]:
        with self._lock:
            now = time.time()
            out: Dict[str, str] = {}
            for r in self._roads:
                out[r] = "ON" if now < self._hold_until.get(r, 0.0) else "OFF"
            return out

    def get_roads(self) -> List[str]:
        with self._lock:
            return list(self._roads)
