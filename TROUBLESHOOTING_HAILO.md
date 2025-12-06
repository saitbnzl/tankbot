# Hailo-8 Object Detection Troubleshooting Guide

This guide helps diagnose and fix issues with Hailo-8 object detection in the tankbot project.

## Problem: No frames in output video stream / Code hangs

### Symptoms
- Video stream shows no frames
- Application appears to hang
- No error logs printed to console

### Recent Fixes Applied

#### 1. Added Comprehensive Logging
All critical operations now have detailed logging with `flush=True` to ensure real-time output:
- Hailo device initialization steps
- Inference pipeline operations
- Detection processing

To enable/disable verbose inference logging, modify `DEBUG_INFERENCE` in `hailo_runner.py`:
```python
DEBUG_INFERENCE = True  # Set to False to reduce log verbosity
```

#### 2. Added Timeout Mechanism
Detection calls now have a configurable timeout (default: 10 seconds) to prevent indefinite hanging:
```python
model = Detector(
    hef_path="resources/yolov8s.hef",
    labels_path="resources/coco_labels.txt",
    config_data=config_data,
    use_hailo=True,
    timeout=10.0,  # Adjust as needed
)
```

If detection times out, the frame is skipped and processing continues.

#### 3. Added Error Handling
Try-catch blocks around all Hailo operations to catch and log errors:
- Initialization errors
- Inference errors
- Post-processing errors

#### 4. Fixed Code Issues
- Removed duplicate variable declarations
- Removed duplicate function definitions
- Added proper exception handling

### Debugging Steps

#### Step 1: Check Logs
Run the server and monitor the console output. You should see:
```
[HAILO] Initializing Hailo with HEF: resources/yolov8s.hef
[HAILO] Loading HEF file...
[HAILO] HEF loaded successfully
[HAILO] Opening VDevice...
[HAILO] VDevice opened successfully
...
[HAILO] Initialization complete!
```

If initialization hangs at a specific step, the logs will show where it stopped.

#### Step 2: Check Hailo Device
Verify the Hailo-8 device is properly connected and recognized:
```bash
# Check if Hailo device is detected
lspci | grep -i hailo

# Check Hailo driver status
dmesg | grep -i hailo
```

#### Step 3: Test with Smaller Model
If yolov8s.hef is too large or causing issues, try a smaller model:
```python
# In tankbot_brain_server.py, change:
hef_path="resources/yolov8s.hef",  # Try yolov8m.hef or smaller
```

#### Step 4: Adjust Timeout
If detection is timing out but should work, increase the timeout:
```python
timeout=30.0,  # Increase from default 10 seconds
```

#### Step 5: Check Resource Usage
Monitor CPU, memory, and Hailo device utilization:
```bash
# Monitor system resources
top

# Check Hailo device temperature/status (if tools available)
hailortcli fw-control identify
```

### Common Issues and Solutions

#### Issue: TimeoutError during inference
**Solution**: 
1. Increase timeout value in `Detector` initialization
2. Check if Hailo device is overheating
3. Verify HEF file is not corrupted
4. Try with smaller input image size (reduce `IMG_SIZE` in config)

#### Issue: "Cannot open VDevice"
**Solution**:
1. Check Hailo driver is loaded: `lsmod | grep hailo`
2. Check device permissions
3. Restart Hailo service if available
4. Reboot system

#### Issue: Slow inference (but not timing out)
**Solution**:
1. Reduce input image size in config
2. Use smaller model (yolov8n instead of yolov8s)
3. Check if other processes are using the Hailo device

#### Issue: Memory errors during initialization
**Solution**:
1. Check available RAM
2. Close other applications
3. Try smaller HEF model
4. Reduce batch size (currently 1, should be minimal)

### Fallback to CPU Inference

If Hailo continues to have issues, you can temporarily switch to CPU-based YOLO:

In `tankbot_brain_server.py` and `person_follow.py`:
```python
USE_HAILO = False  # Switch to CPU YOLO
```

This will use Ultralytics YOLO on CPU instead of Hailo, which is slower but more stable for debugging.

### Getting More Help

If issues persist:
1. Check Hailo community forums
2. Review Hailo-8 documentation
3. Verify HEF model compatibility with your Hailo firmware version
4. Check GitHub issues for similar problems

### Configuration Reference

Key configuration files:
- `hailo_runner.py` - Hailo inference configuration
- `detector.py` - Detector wrapper with timeout
- `person_follow_config.py` - Person tracking parameters
- `tankbot_brain_server.py` - Server configuration

Key parameters:
- `DEBUG_INFERENCE` - Enable/disable verbose logging
- `timeout` - Detection timeout in seconds
- `IMG_SIZE` - Input image size (affects performance)
- `HEF_PATH` - Path to Hailo model file

### Performance Tuning

For optimal performance:
1. Set `DEBUG_INFERENCE = False` after initial debugging
2. Adjust `IMG_SIZE` based on accuracy vs speed tradeoff
3. Monitor frame processing time and adjust timeout accordingly
4. Consider using smaller models for real-time performance

### Log Markers

Key log prefixes for debugging:
- `[HAILO]` - Hailo device and inference operations
- `[DETECTOR]` - Detector wrapper operations (timeout, errors)
- `[FOLLOW]` - Person following loop operations
- `[SERVER]` - Web server and video streaming operations
- `[ERROR]` - Error messages requiring attention
