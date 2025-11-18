# tankbot_brain_server.py

from fastapi.responses import StreamingResponse, FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO
import asyncio
import websockets
import cv2
import threading
import time
import json


from person_follow import person_follow_loop
from person_follow_config import get_config, update_config

person_follow_thread = None
person_follow_stop_event: threading.Event | None = None


# ============================================================
#               GLOBAL SETTINGS
# ============================================================

VIDEO_URL = "http://192.168.1.50:81/stream"
WS_URL    = "ws://tankbot.local:81"

model = YOLO("yolov8n.pt")
app = FastAPI()

person_follow_task: asyncio.Task | None = None
person_follow_stop_event: asyncio.Event | None = None

# last command for UI
last_sent_command = {"cmd": "stop", "speed": 0}


# ============================================================
#             GLOBAL FRAME GRABBER (single stream)
# ============================================================

latest_frame = None
frame_lock = threading.Lock()


def frame_grabber():
    global latest_frame
    print("[FRAME] Starting frame grabber...")

    cap = cv2.VideoCapture(VIDEO_URL)
    if not cap.isOpened():
        raise RuntimeError("Cannot open ESP32-CAM stream")

    while True:
        ok, frame = cap.read()
        if ok and frame is not None:
            # Rotate if your ESP32-CAM orientation requires it
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            with frame_lock:
                latest_frame = frame
        time.sleep(0.001)


threading.Thread(target=frame_grabber, daemon=True).start()


def get_latest_frame():
    with frame_lock:
        if latest_frame is None:
            raise RuntimeError("No frame yet")
        return latest_frame.copy()


def annotate_frame(frame):
    results = model(frame, imgsz=320, verbose=False)
    return results[0].plot()


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

@app.get("/detect")
def detect():
    frame = get_latest_frame()
    results = model(frame, imgsz=320, verbose=False)[0]

    detections = []
    if results.boxes is not None:
        for box, cid, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = map(float, box)
            detections.append({
                "class_id": int(cid),
                "class_name": model.names[int(cid)],
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2]
            })

    return {"count": len(detections), "detections": detections}


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

    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/status")
def status():
    return {
        "last_command": last_sent_command,
        "latest_frame": "ok" if latest_frame is not None else "none"
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
                async with websockets.connect(WS_URL) as ws:
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
