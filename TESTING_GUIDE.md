# Testing Guide for Hailo-8 Object Detection Fixes

This guide helps you test the fixes for the Hailo-8 object detection hanging issues.

## Quick Start Testing

### 1. Test with Debug Logging Enabled

First, test with debug logging to see all diagnostic output:

```bash
# Enable debug logging
export HAILO_DEBUG_INFERENCE=1

# Start the server
python tankbot_brain_server.py
```

**What to look for:**
- You should see detailed `[HAILO]` initialization messages
- Watch for where the initialization process stops if it hangs
- Note any error messages or timeouts

### 2. Test with Debug Logging Disabled (Production Mode)

After confirming it works, test in production mode:

```bash
# Disable debug logging (default)
unset HAILO_DEBUG_INFERENCE

# Or explicitly set to 0
export HAILO_DEBUG_INFERENCE=0

# Start the server
python tankbot_brain_server.py
```

**What to look for:**
- Less verbose output (only errors and important events)
- System should run more efficiently
- Errors still logged if they occur

## Testing Scenarios

### Scenario 1: Verify Hailo Initialization

**Expected Behavior:**
```
[HAILO] Initializing Hailo with HEF: resources/yolov8s.hef
[HAILO] Loading HEF file...
[HAILO] HEF loaded successfully
[HAILO] Opening VDevice...
[HAILO] VDevice opened successfully
[HAILO] Configuring device with HEF...
[HAILO] Device configured successfully
[HAILO] Getting stream infos...
[HAILO] Input shape: (640, 640, 3)
[HAILO] Creating vstream params...
[HAILO] Vstream params created successfully
[HAILO] Loading labels from: resources/coco_labels.txt
[HAILO] Loaded 80 labels
[HAILO] Initialization complete!
```

**If it hangs:**
- The last logged message shows where initialization stopped
- Check the troubleshooting guide for that specific step
- Common issues:
  - "Opening VDevice" → Check Hailo driver/device
  - "Configuring device" → Check HEF file compatibility
  - "Loading HEF file" → Check file path and permissions

### Scenario 2: Verify Video Stream Works

**Steps:**
1. Start the server
2. Open browser to `http://localhost:8000`
3. Check if video stream appears

**Expected Behavior:**
- Video stream shows frames (with or without detections)
- If detection fails, frames show without annotations
- No complete freeze of the stream

**If problems occur:**
- Check console for `[SERVER][ERROR]` messages
- TimeoutError messages indicate detection is timing out
- Stream should continue showing frames even if detection fails

### Scenario 3: Test Person Following Mode

**Steps:**
1. Start the server
2. Click "Start Person Following" in web UI
3. Stand in front of camera

**Expected Behavior:**
- System detects person and tries to follow
- If detection times out, system continues trying
- Motor commands sent based on person position

**What to monitor:**
```
[FOLLOW] Detection returned X results
[FOLLOW][TRACK] forward err=0.15 conf=0.85 ...
```

**If problems occur:**
```
[FOLLOW][ERROR] Detection timed out: Detection timed out after 10.0 seconds
[FOLLOW] Skipping this frame and continuing...
```
This is normal - system will retry on next frame.

### Scenario 4: Test Timeout Mechanism

**Steps:**
1. If you have a known problematic scenario, test it
2. Watch for timeout messages
3. Verify system continues after timeout

**Expected Behavior:**
```
[DETECTOR][ERROR] Detection timed out after 10.0 seconds!
[DETECTOR][WARNING] Background thread will complete eventually
```

System should:
- Not hang completely
- Skip the problematic frame
- Continue with next frame after 0.5 seconds

## Configuration Testing

### Adjust Timeout Value

If detection consistently times out but you think it should work:

Edit `tankbot_brain_server.py`:
```python
model = Detector(
    hef_path="resources/yolov8s.hef",
    labels_path="resources/coco_labels.txt",
    config_data=config_data,
    use_hailo=True,
    timeout=30.0,  # Increase from 10 to 30 seconds
)
```

### Test CPU Fallback

If Hailo continues to have issues, test CPU mode:

Edit `tankbot_brain_server.py` and `person_follow.py`:
```python
USE_HAILO = False  # Switch to CPU YOLO
```

## Performance Metrics

### Check Detection Speed

With debug logging enabled, time between these messages:
```
[HAILO] Starting inference...
...
[HAILO] Inference successful, found X detections
```

**Typical times:**
- Good: 50-200ms per frame
- Acceptable: 200-500ms per frame
- Slow: 500ms-2s per frame
- Problem: >2s per frame (may timeout)

### Monitor System Resources

```bash
# Check CPU and memory
top

# Check Hailo device (if tools available)
hailortcli fw-control identify
```

## Troubleshooting During Testing

### Problem: "No module named 'hailo'"

**Solution:**
```bash
# Install Hailo Python bindings
bash hailo_python_installation.sh

# Or install manually
pip install hailo-platform
```

### Problem: "Cannot open VDevice"

**Solution:**
```bash
# Check if Hailo device is detected
lspci | grep -i hailo

# Check driver
lsmod | grep hailo

# May need to restart Hailo service or reboot
```

### Problem: Constant TimeoutErrors

**Solution:**
1. Increase timeout value (see Configuration Testing above)
2. Reduce image size in config:
   ```python
   "IMG_SIZE": 224,  # Reduce from 320
   ```
3. Try smaller model (yolov8n instead of yolov8s)
4. Check system resources (CPU, memory)

### Problem: High CPU Usage

**Solution:**
1. Disable debug logging:
   ```bash
   export HAILO_DEBUG_INFERENCE=0
   ```
2. Reduce detection frequency:
   ```python
   "TRACK_INTERVAL_SEC": 0.3,  # Increase from 0.15
   ```

## Logging Configuration

### Enable Debug Logging for Specific Session

```bash
HAILO_DEBUG_INFERENCE=1 python tankbot_brain_server.py
```

### Permanently Enable Debug Logging

Add to your `.bashrc` or `.profile`:
```bash
export HAILO_DEBUG_INFERENCE=1
```

### Redirect Logs to File

```bash
python tankbot_brain_server.py 2>&1 | tee tankbot.log
```

This saves all output to `tankbot.log` while showing it on screen.

## Success Criteria

The fixes are working correctly if:

1. ✓ System doesn't hang indefinitely
2. ✓ Detailed logs show where any issues occur
3. ✓ TimeoutError is raised after 10 seconds (or configured timeout)
4. ✓ System continues operation after timeout
5. ✓ Video stream shows frames even when detection fails
6. ✓ Person following continues attempting detection after errors

## Reporting Issues

If you still encounter problems after testing:

1. Collect the following information:
   - Full console output with `HAILO_DEBUG_INFERENCE=1`
   - System specs (CPU, RAM, Hailo device model)
   - Hailo driver/firmware version
   - Python version and package versions

2. Note at which step the system hangs (from logs)

3. Try the CPU fallback mode to determine if it's Hailo-specific

4. Check the TROUBLESHOOTING_HAILO.md guide

5. Open an issue with collected information

## Next Steps

After successful testing:

1. Set `HAILO_DEBUG_INFERENCE=0` for production
2. Tune timeout based on your hardware performance
3. Adjust IMG_SIZE and other parameters for optimal performance
4. Review TROUBLESHOOTING_HAILO.md for optimization tips
5. Monitor logs periodically for any recurring errors
