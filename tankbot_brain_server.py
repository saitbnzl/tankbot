# tankbot_brain_server.py

from fastapi.responses import StreamingResponse, FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import websockets
import cv2
import threading
import time
import json
from hailo_runner import _run_hailo

from person_follow import person_follow_loop
from person_follow_config import get_config, update_config
import json

with open("resources/yolo_conf.json") as f:
    config_data = json.load(f)

person_follow_thread = None
person_follow_stop_event: threading.Event | None = None

_latest_frame = None
_latest_frame_lock = threading.Lock()

# NEW: toggle between CPU YOLO and Hailo
USE_HAILO = True  # set False to go back to plain Ultralytics on CPU

if USE_HAILO:
    # NEW: Hailo-backed YOLO wrapper which mimics Ultralytics API
    from detector import Detector
else:
    from ultralytics import YOLO as YoloDetector

from person_follow import person_follow_loop
from person_follow_config import get_config, update_config

person_follow_thread = None
person_follow_stop_event: threading.Event | None = None


# ============================================================
#               GLOBAL SETTINGS
# ============================================================

VIDEO_URL = "http://192.168.1.50:81/stream"
WS_URL    = "ws://tankbot.local:81"

# Error recovery delay for video streaming (seconds)
ERROR_RECOVERY_DELAY = 0.1
# Delay between frame grabber reconnect attempts
FRAME_RECONNECT_DELAY = 1.0

if USE_HAILO:
    model = Detector(
        hef_path="resources/yolov8s.hef",
        labels_path="resources/coco_labels.txt",
        config_data=config_data,
        # class_filter=[0],
        use_hailo=True,
        timeout=10.0,  # 10 second timeout for Hailo inference
    )
else:
    model = YoloDetector("yolov8n.pt")

app = FastAPI()

person_follow_task: asyncio.Task | None = None
person_follow_stop_event: asyncio.Event | None = None

# last command for UI
last_sent_command = {"cmd": "stop", "speed": 0}


# ============================================================
#             GLOBAL FRAME GRABBER (single stream)
# ============================================================


def frame_grabber():
    print("[FRAME] Starting frame grabber...")
    cap = None

    while True:
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            print(f"[FRAME] Connecting to stream: {VIDEO_URL}", flush=True)
            cap = cv2.VideoCapture(VIDEO_URL)
            if not cap.isOpened():
                print("[FRAME][WARN] Failed to open stream, retrying...", flush=True)
                cap.release()
                cap = None
                time.sleep(FRAME_RECONNECT_DELAY)
                continue
            print("[FRAME] Stream opened", flush=True)
        ok, frame = cap.read()
        if ok and frame is not None:
            # Rotate if your ESP32-CAM orientation requires it
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            update_latest_frame(frame)
        else:
            print("[FRAME][WARN] Frame grab failed, forcing reconnect", flush=True)
            cap.release()
            cap = None
            time.sleep(FRAME_RECONNECT_DELAY)
        time.sleep(0.001)


threading.Thread(target=frame_grabber, daemon=True).start()


def get_latest_frame(block=True, sleep_time=0.01):
    """
    Returns the latest frame. If block=True, waits until a frame is available.
    """
    global _latest_frame
    if not block:
        if _latest_frame is None:
            return None
        with _latest_frame_lock:
            return _latest_frame

    # block=True
    while True:
        with _latest_frame_lock:
            if _latest_frame is not None:
                return _latest_frame
        time.sleep(sleep_time)


def update_latest_frame(frame):
    global _latest_frame
    with _latest_frame_lock:
        _latest_frame = frame


def annotate_frame(frame):
    """
    Returns an annotated frame for streaming.
    Uses unified Detector (Hailo or YOLO) which returns a list of detections:
      {
        "class_id": int,
        "class_name": str,
        "confidence": float,
        "bbox": [x1, y1, x2, y2],
      }
    """
    # İstersen IMG_SIZE config'ten gelebilir; şimdilik sabit varsayalım:
    IMG_SIZE = 320

    try:
        detections = model(frame, imgsz=IMG_SIZE, verbose=False)
    except TimeoutError as e:
        print(f"[SERVER][ERROR] Detection timed out in annotate_frame: {e}", flush=True)
        # Return unannotated frame on timeout
        return frame
    except Exception as e:
        print(f"[SERVER][ERROR] Detection failed in annotate_frame: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Return unannotated frame on error
        return frame

    annotated = frame.copy()
    for det in detections:
        cid = int(det["class_id"])
        # sadece person çizmek istersen:
        if cid != 0:
            continue

        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        label = det["class_name"]

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        txt = f"{label} {conf:.2f}"
        cv2.putText(annotated, txt, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, txt, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


# ============================================================
#         SHARED MOTOR CONTROL FUNCTION (IMPORTANT)
# ============================================================

async def send_motor_command(cmd: str, speed: int):
    """
    Unified motor sender for both /drive and person-follow mode.
    Ensures last_sent_command is ALWAYS updated.
    """
    global last_sent_command
    msg = f"{cmd},{speed}"


    async with websockets.connect(WS_URL) as ws:
        await ws.send(msg)
        print(f"[MOTOR][SEND] {msg}", flush=True)


    last_sent_command = {"cmd": cmd, "speed": speed}


# ============================================================
#                     DATA MODELS
# ============================================================

class DriveRequest(BaseModel):
    cmd: str
    speed: int = 70


# ============================================================
#                     ROUTES
# ============================================================


@app.post("/drive")
async def drive(req: DriveRequest):
    cmd = req.cmd.lower()

    # normalize
    if cmd in ("s", "stp"):
        cmd = "stop"
    if cmd == "f":
        cmd = "forward"
    if cmd == "b":
        cmd = "backward"
    if cmd == "l":
        cmd = "left"
    if cmd == "r":
        cmd = "right"

    if cmd not in ("forward", "backward", "left", "right", "stop"):
        raise HTTPException(status_code=400, detail="Invalid cmd")

    try:
        await send_motor_command(cmd, req.speed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok"}


@app.get("/video")
def video():
    def generator():
        while True:
            try:
                frame = get_latest_frame()
                annotated = annotate_frame(frame)
                ok, jpg = cv2.imencode(".jpg", annotated)
                if not ok:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    jpg.tobytes() +
                    b"\r\n"
                )
            except Exception as e:
                print(f"[SERVER][ERROR] Video generator error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                import time
                time.sleep(ERROR_RECOVERY_DELAY)  # avoid tight loop on persistent errors

    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/status")
def status():
    return {
        "last_command": last_sent_command,
        "latest_frame": "ok" if get_latest_frame(block=False) is not None else "none"
    }


@app.post("/person_follow/start")
async def person_follow_start():
    global person_follow_thread, person_follow_stop_event

    if person_follow_thread is not None and person_follow_thread.is_alive():
        raise HTTPException(status_code=400, detail="Person follow already running")

    if not isinstance(WS_URL, str):
        raise HTTPException(status_code=500, detail=f"WS_URL is corrupted: {type(WS_URL)}")

    # new stop event for this run
    person_follow_stop_event = threading.Event()

    print("[DEBUG] Starting person-follow THREAD", flush=True)

    def runner():
        async def main():
            # 1) WS bağlantısını AÇ
            print(f"[FOLLOW] Connecting motor WS: {WS_URL}", flush=True)
            try:
                async with websockets.connect(
                    WS_URL,
                    ping_interval=None,
                    ping_timeout=None,
                ) as ws:
                    print("[FOLLOW] Motor WS connected", flush=True)

                    # 2) Bu WS üzerinden komut gönderen fonksiyon
                    async def send_motor_command(cmd: str, speed: int):
                        msg = f"{cmd},{speed}"
                        await ws.send(msg)

                    # 3) Follow loop'u başlat
                    await person_follow_loop(
                        get_latest_frame,          # frame provider
                        send_motor_command,        # <<< FONKSİYON, string değil
                        model,
                        person_follow_stop_event,  # threading.Event ama is_set() yetiyor
                    )

            except Exception as e:
                print("[FOLLOW][WS ERROR]", repr(e), flush=True)
                await asyncio.sleep(1)  # 1 sn sonra reconnect dene


        try:
            asyncio.run(main())
        except Exception as e:
            print("[FOLLOW][THREAD ERROR]", repr(e), flush=True)

    person_follow_thread = threading.Thread(target=runner, daemon=True)
    person_follow_thread.start()

    return {"status": "started"}

@app.post("/person_follow/stop")
async def person_follow_stop():
    global person_follow_thread, person_follow_stop_event

    if person_follow_stop_event is not None:
        person_follow_stop_event.set()
        print("[DEBUG] person_follow_stop_event set", flush=True)

    try:
        await send_motor_command("stop", 0)
    except Exception:
        pass

    # thread kendisi bitecek; sadece referansı bırak
    person_follow_thread = None
    person_follow_stop_event = None

    return {"status": "stopped"}



@app.get("/config")
def config_get():
    return get_config()

class UpdateConfigRequest(BaseModel):
    key: str
    value: float | int | bool | str

@app.post("/config/update")
def config_update(req: UpdateConfigRequest):
    new_cfg = update_config({req.key: req.value})
    return new_cfg

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")
