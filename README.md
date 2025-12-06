# TankBot - AI-Powered Tank Robot with Hailo-8 Acceleration

A person-following robot using object detection with Hailo-8 hardware acceleration or CPU-based YOLO inference.

## Features

- **Person Detection & Tracking**: Automatically detects and follows a person
- **Hailo-8 Hardware Acceleration**: Fast inference using Hailo-8 AI accelerator
- **Fallback CPU Mode**: Can run on CPU using Ultralytics YOLO if Hailo is unavailable
- **Real-time Video Streaming**: Live video feed with object detection annotations
- **Adaptive Movement**: Smart turning and following behavior with obstacle detection
- **Web Interface**: Control and monitor the robot through a web browser
- **RESTful API**: Control robot movements programmatically

## Hardware Requirements

- Hailo-8 AI Accelerator (optional, can run on CPU)
- ESP32-CAM or compatible camera module
- Motor controller compatible with the tankbot
- Sufficient computing power (Raspberry Pi 4 or similar)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Hailo Software (Optional)

If using Hailo-8 acceleration:

```bash
bash hailo_python_installation.sh
```

Follow the Hailo documentation for driver installation.

### 3. Configure Video Source

Edit `tankbot_brain_server.py` to set your ESP32-CAM URL:

```python
VIDEO_URL = "http://192.168.1.50:81/stream"  # Your ESP32-CAM IP
WS_URL = "ws://tankbot.local:81"             # Your motor controller WebSocket
```

## Usage

### Start the Server

```bash
bash start_server.sh
```

Or manually:

```bash
python tankbot_brain_server.py
```

The server will start on `http://localhost:8000`

### Web Interface

Open your browser and navigate to `http://localhost:8000`

Features:
- Live video feed with person detection
- Manual control buttons (forward, backward, left, right, stop)
- Person following mode toggle
- Configuration panel for tuning parameters

### API Endpoints

#### Manual Control
```bash
# Drive forward
curl -X POST http://localhost:8000/drive -H "Content-Type: application/json" -d '{"cmd": "forward", "speed": 70}'

# Stop
curl -X POST http://localhost:8000/drive -H "Content-Type: application/json" -d '{"cmd": "stop", "speed": 0}'
```

#### Person Following
```bash
# Start person following mode
curl -X POST http://localhost:8000/person_follow/start

# Stop person following mode
curl -X POST http://localhost:8000/person_follow/stop
```

#### Configuration
```bash
# Get current configuration
curl http://localhost:8000/config

# Update configuration
curl -X POST http://localhost:8000/config/update -H "Content-Type: application/json" -d '{"key": "TURN_SPEED", "value": 50}'
```

#### Status
```bash
# Check system status
curl http://localhost:8000/status
```

## Configuration

### Person Following Parameters

Edit `person_follow_config.py` or use the web interface:

```python
config = {
    "CONF_THRESHOLD": 0.50,      # Person detection confidence threshold
    "IMG_SIZE": 320,             # Image size for detection (higher = better accuracy, slower)
    "CENTER_DEADZONE": 0.21,     # Center alignment tolerance (±21%)
    "TURN_SPEED": 25,            # Rotation speed when tracking
    "FORWARD_SPEED": 70,         # Forward movement speed
    "SEARCH_TURN_SPEED": 50,     # Speed when searching for person
    "LOST_FRAMES_LIMIT": 200,    # Frames before giving up search
    "LOST_FRAMES_GRACE": 15,     # Frames before starting search
    "TRACK_INTERVAL_SEC": 0.15,  # Delay between control commands
    "MAX_PERSON_AREA": 0.70,     # Stop if person is too close (70% of frame)
    "STOP_ON_TOO_CLOSE": True,   # Enable proximity stopping
}
```

### Switching Between Hailo and CPU

In `tankbot_brain_server.py` and `person_follow.py`:

```python
USE_HAILO = True   # Use Hailo-8 hardware acceleration
USE_HAILO = False  # Use CPU-based YOLO (slower but more compatible)
```

### Detection Timeout

In `tankbot_brain_server.py`:

```python
model = Detector(
    hef_path="resources/yolov8s.hef",
    labels_path="resources/coco_labels.txt",
    config_data=config_data,
    use_hailo=True,
    timeout=10.0,  # Timeout in seconds (adjust based on your hardware)
)
```

## Troubleshooting

If you experience issues with Hailo-8 detection (no frames, hanging, etc.), see the [Hailo Troubleshooting Guide](TROUBLESHOOTING_HAILO.md) for detailed debugging steps.

### Common Issues

**No video frames appearing:**
- Check camera connection and URL
- Review logs for timeout or error messages
- Try increasing the timeout value
- Switch to CPU mode temporarily

**Detection timing out:**
- Increase timeout in Detector initialization
- Check Hailo device status
- Verify HEF model file is not corrupted
- Try smaller image size

**High CPU usage:**
- Reduce IMG_SIZE in configuration
- Set DEBUG_INFERENCE = False in hailo_runner.py
- Use smaller YOLO model

## Architecture

```
┌─────────────────┐
│  Web Interface  │
│   (Browser)     │
└────────┬────────┘
         │ HTTP/WebSocket
         ▼
┌─────────────────────────┐
│  FastAPI Server         │
│  (tankbot_brain_server) │
└────┬───────────────┬────┘
     │               │
     ▼               ▼
┌─────────┐    ┌──────────────┐
│ Camera  │    │ Motor Control│
│ ESP32   │    │  WebSocket   │
└────┬────┘    └──────────────┘
     │
     ▼
┌──────────────────────┐
│   Detector           │
│   (Hailo or CPU)     │
└──────────────────────┘
     │
     ├─► Hailo-8 Device
     └─► CPU YOLO
```

## File Structure

```
tankbot/
├── hailo_runner.py              # Hailo-8 inference wrapper
├── detector.py                  # Unified detector (Hailo/CPU)
├── person_follow.py             # Person following logic
├── person_follow_config.py      # Configuration parameters
├── tankbot_brain_server.py      # Main FastAPI server
├── object_detection_post_process.py  # Detection post-processing
├── resources/
│   ├── yolov8s.hef             # Hailo model file
│   ├── yolov8m.hef             # Alternative model
│   ├── coco_labels.txt         # Class labels
│   └── yolo_conf.json          # YOLO configuration
├── static/
│   └── index.html              # Web interface
├── common/                     # Utility modules
│   ├── hailo_inference.py
│   ├── toolbox.py
│   └── tracker/                # ByteTrack tracker
└── requirements.txt
```

## Development

### Adding Custom Behaviors

Extend `person_follow.py` to add custom robot behaviors:

```python
# Example: Add emergency stop on specific gesture
if detect_stop_gesture(frame):
    await send_motor_command("stop", 0)
    state = "IDLE"
```

### Custom Detection Models

To use a different Hailo model:

1. Place your `.hef` file in `resources/`
2. Update the path in server initialization:
   ```python
   hef_path="resources/your_model.hef"
   ```
3. Ensure labels file matches your model's classes

### Logging

Control log verbosity:
- Set `DEBUG_INFERENCE = False` in `hailo_runner.py` for production
- All logs use `flush=True` for real-time output
- Look for `[HAILO]`, `[DETECTOR]`, `[FOLLOW]`, `[SERVER]` prefixes

## Performance Tips

1. **Image Size**: Start with 320, increase for better accuracy
2. **Model Selection**: yolov8n (fastest) → yolov8s → yolov8m (most accurate)
3. **Frame Rate**: Adjust TRACK_INTERVAL_SEC based on your needs
4. **Debug Mode**: Disable DEBUG_INFERENCE for production use

## Contributing

Contributions are welcome! Please:
1. Test changes with both Hailo and CPU modes
2. Update documentation for new features
3. Add error handling and logging
4. Follow existing code style

## License

[Add your license here]

## Credits

- YOLOv8 by Ultralytics
- Hailo-8 AI Accelerator
- ByteTrack for object tracking
- FastAPI web framework

## Support

For issues and questions:
1. Check [TROUBLESHOOTING_HAILO.md](TROUBLESHOOTING_HAILO.md)
2. Review console logs for error messages
3. Open an issue on GitHub with logs and system info
