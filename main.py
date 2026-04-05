import os
import threading
import time
from typing import Dict, Optional
import sys
import json
try:
    from flask import Flask, jsonify, send_from_directory
    from flask_cors import CORS
except ImportError:
    Flask = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    from detection.vehicle_detector import VehicleDetector, DetectionResult
except Exception:
    from dataclasses import dataclass
    from typing import Optional, Tuple
    try:
        from detection.light_detector import EmergencyLightDetector
    except Exception:
        EmergencyLightDetector = None

    @dataclass
    class DetectionResult:
        detected: bool
        confidence: float
        bounding_box: Optional[Tuple[int, int, int, int]]
        road_direction: str

    class VehicleDetector:
        def __init__(self, confidence_threshold: float = 0.85) -> None:
            self.confidence_threshold = confidence_threshold
            self.light_detector = EmergencyLightDetector() if EmergencyLightDetector else None

        def detect(self, frame, road_direction: str) -> DetectionResult:
            if self.light_detector:
                detected, conf = self.light_detector.detect_flashing(frame, road_direction)
                return DetectionResult(detected, conf, None, road_direction)
            return DetectionResult(False, 0.0, None, road_direction)
from emergency_lights.light_controller import LightController
from emergency_lights.road_monitor import RoadMonitor


def load_config(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not os.path.exists(path):
        data["north"] = ""
        data["south"] = ""
        data["east"] = ""
        data["west"] = ""
        data["confidence_threshold"] = "0.3"
        return data
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            key = k.strip()
            val = v.strip()
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            if val.startswith("'") and val.endswith("'"):
                val = val[1:-1]
            data[key] = val
    for alt, canon in [
        ("north_url", "north"),
        ("south_url", "south"),
        ("east_url", "east"),
        ("west_url", "west"),
    ]:
        if alt in data and canon not in data:
            data[canon] = data[alt]
    if "confidence_threshold" not in data:
        data["confidence_threshold"] = "0.3"
    return data


class CameraWorker(threading.Thread):
    def __init__(
        self,
        name: str,
        url: str,
        detector: VehicleDetector,
        monitor: RoadMonitor,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.name = name
        self.url = url
        self.detector = detector
        self.monitor = monitor
        self.stop_event = stop_event
        self._cap = None

    def _open(self) -> bool:
        if cv2 is None:
            return False
        try:
            # Handle webcam index if URL is a digit string
            source = self.url
            if isinstance(source, str) and source.isdigit():
                source = int(source)
            self._cap = cv2.VideoCapture(source)
            return bool(self._cap is not None and self._cap.isOpened())
        except Exception:
            return False

    def _release(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None

    def run(self) -> None:
        if not self.url:
            print(f"{self.name} camera URL not provided. Using simulation mode.")
            return
        retry_delay = 2.0
        while not self.stop_event.is_set():
            print(f"[{self.name}] Attempting to connect to {self.url}...")
            if not self._open():
                print(f"[{self.name}] Failed to connect. Retrying in {int(retry_delay)}s.")
                time.sleep(retry_delay)
                continue
            
            print(f"[{self.name}] Connected successfully.")
            try:
                while not self.stop_event.is_set():
                    ok, frame = self._cap.read()
                    if not ok or frame is None:
                        print(f"[{self.name}] Stream lost. Reconnecting in {int(retry_delay)}s.")
                        break
                    result: DetectionResult = self.detector.detect(frame, self.name)
                    self.monitor.update(self.name, result.detected, result.confidence)
                    time.sleep(0.01) # Reduced sleep for better responsiveness
            finally:
                self._release()
                time.sleep(retry_delay)


# North-only demo is no longer active. Restoring all roads.
def main() -> None:
    cfg = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
    roads = ["North", "South", "East", "West"]
    urls = {
        "North": cfg.get("north", ""),
        "South": cfg.get("south", ""),
        "East": cfg.get("east", ""),
        "West": cfg.get("west", ""),
    }
    try:
        conf_thr = float(cfg.get("confidence_threshold", "0.85"))
    except Exception:
        conf_thr = 0.85

    controller = LightController(roads)
    monitor = RoadMonitor(controller)
    detector = VehicleDetector(confidence_threshold=conf_thr)

    stop_event = threading.Event()
    workers = {}  # Use a dict to track workers by road name

    def start_web_server(controller: LightController, urls_dict: Dict[str, str], workers_dict: Dict[str, threading.Thread]):
        if Flask is None:
            print("Flask not installed. Web dashboard will not be available.")
            return
        
        app = Flask(__name__)
        CORS(app)
        
        @app.route('/api/status')
        def get_status():
            # Include which roads have URLs (are "connected")
            connected_roads = {r: bool(urls_dict.get(r)) for r in roads}
            return jsonify({
                "active_roads": controller.get_active_roads(),
                "all_states": controller.get_all_states(),
                "connected_roads": connected_roads,
                "urls": urls_dict,
                "timestamp": time.time(),
                "system_healthy": True
            })
        
        from flask import request
        @app.route('/api/camera', methods=['POST'])
        def add_camera():
            data = request.json
            road = data.get('road')
            url = data.get('url')
            if road not in roads:
                return jsonify({"error": "Invalid road"}), 400
            
            # Update or start the worker
            if road in workers_dict:
                print(f"Updating camera for {road}...")
                workers_dict[road].url = url
            else:
                w = CameraWorker(road, url, detector, monitor, stop_event)
                workers_dict[road] = w
                w.start()
                print(f"Started camera worker for {road} with URL: {url}")
            
            urls_dict[road] = url
            save_config(road.lower(), url)
            return jsonify({"status": "success", "road": road, "url": url})

        @app.route('/api/spawn', methods=['POST'])
        def spawn_ambulance():
            data = request.json
            road = data.get('road')
            if road not in roads:
                return jsonify({"error": "Invalid road"}), 400
            
            # Simulate an ambulance detection on this road
            # This triggers the 30-second hold duration in the LightController
            controller.turn_on(road)
            print(f"Manual simulation: Ambulance spawned on {road}")
            return jsonify({"status": "success", "road": road})

        @app.route('/')
        def index():
            return send_from_directory('static', 'index.html')

        @app.route('/static/<path:path>')
        def send_static(path):
            return send_from_directory('static', path)

        try:
            app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
        except Exception as e:
            print(f"Could not start web server: {e}")

    def save_config(key: str, value: str):
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        lines = []
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        
        found = False
        new_lines = []
        for line in lines:
            if line.strip().startswith(f"{key}:"):
                new_lines.append(f"{key}: \"{value}\"\n")
                found = True
            else:
                new_lines.append(line)
        
        if not found:
            new_lines.append(f"{key}: \"{value}\"\n")
            
        with open(config_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    # Start Web Dashboard server in a background thread
    web_thread = threading.Thread(target=start_web_server, args=(controller, urls, workers), daemon=True)
    web_thread.start()
    print("\n" + "="*50)
    print("TRAFFIC CONTROL SYSTEM ACTIVE")
    print(f"Dashboard: http://localhost:5000/")
    print("="*50 + "\n")

    # Start workers only for configured URLs initially
    for r, url in urls.items():
        if url and len(url) > 1:
            w = CameraWorker(r, url, detector, monitor, stop_event)
            workers[r] = w
            w.start()

    print("Running in headless mode. All control and simulation are via the web dashboard.")
    try:
        # Keep the main thread alive for the web server and camera workers
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_event.set()
        print("Stopping camera workers...")
        for w in workers.values():
            w.join(timeout=1.0)
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
