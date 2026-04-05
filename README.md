# Smart Traffic Ambulance Detection System

An AI-powered traffic control system that prioritizes emergency vehicles using YOLOv8 classification and real-time monitoring.

## Features
- **Ambulance Detection**: Custom-trained YOLOv8 model specifically for ambulances.
- **Traffic Control**: Automatically prioritizes roads where an ambulance is detected.
- **Web Dashboard**: Real-time monitoring and simulation tools.
- **Dynamic Configuration**: Easily configure camera URLs and detection thresholds.

## Setup

### Prerequisites
- Python 3.9+
- Node.js (for frontend dependencies)

### Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd smart-traffic-ambulance
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install frontend dependencies:
   ```bash
   npm install
   ```

## Running the Application
To start the system:
```bash
python main.py
```
Then open [http://localhost:5000](http://localhost:5000) in your browser.

## Project Structure
- `detection/`: AI detection logic and model wrappers.
- `emergency_lights/`: Traffic light control and monitoring logic.
- `models/`: Trained model weights.
- `static/`: Web dashboard frontend files.
- `main.py`: Main entry point.
