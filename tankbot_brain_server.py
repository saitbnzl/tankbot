# tankbot_brain_server.py

from fastapi.responses import StreamingResponse, FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO
import requests
import asyncio
import websockets
import cv2
import threading
import time

from person_follow import person_follow_loop


# ============================================================
#               GLOBAL SETTINGS
# ============================================================

VIDEO_URL = "http://192.168.1.50:81/stream"  # ESP32-CAM stream
WS_URL    = "ws://tankbot.local:81"          # tankbot motor controller

model = YOLO("yolov8n.pt")

app = FastAPI()

person_follow_task: asyncio.Task | None = None
person_follow_stop_event: asyncio.Event | None = None


# ============================================================
#                 GLOBAL FRAME BUFFER
# ============================================================

latest_frame = None
frame_lock = threading.Lock()

def frame_grabber():
    """
    Runs forever in a dedicated thread.
    Reads frames from ESP32-CAM using a SINGLE VideoCapture.
    Stores the *latest* frame in global latest_frame.
    """
    global latest_frame

    print("[FRAME] Starting frame grabber thread...")
    cap = cv2.VideoCapture(VIDEO_URL)

    if not cap.isOpened():
        raise RuntimeError("Cannot open ESP32-CAM stream")

    while True:
        ok, frame = cap.read()
        if ok and frame is not None:
            # Rotate camera (ESP32-CAM is sideways)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            with frame_lock:
                latest_frame = frame

        # small sleep avoids hogging CPU
        time.sleep(0.001)


# Start frame grabber thread on startup
threading.Thread(target=frame_grabber, daemon=True).start()


# ============================================================
#                UTILITY FUNCTIONS
# ============================================================

def get_latest_frame():
    global latest_frame
    with frame_lock:
        if latest_frame is None:
            raise RuntimeError("No frame received yet")
        return latest_frame.copy()


def annotate_frame(frame):
    results = model(frame, imgsz=320, verbose=False)
    return results[0].plot()


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
    try:
        frame = get_latest_frame()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = model(frame, imgsz=320, verbose=False)[0]

    detections = []
    if results.boxes is not None:
        for box, cid, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = [float(v) for v in box]
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
    if cmd not in ("forward", "backward", "left", "right", "stop", "f", "b", "l", "r", "s"):
        raise HTTPException(status_code=400, detail="Invalid cmd")

    try:
        async with websockets.connect(WS_URL) as ws:
            await ws.send(f"{cmd},{req.speed}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WS error: {e}")

    return {"status": "ok"}


@app.get("/video")
def video():
    def generator():
        while True:
            try:
                frame = get_latest_frame()
                annotated = annotate_frame(frame)
            except:
                continue

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
        "latest_frame": "ok" if latest_frame is not None else "none",
    }


@app.post("/person_follow/start")
async def person_follow_start():
    global person_follow_task, person_follow_stop_event

    if person_follow_task and not person_follow_task.done():
        raise HTTPException(status_code=400, detail="Already running")

    person_follow_stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    person_follow_task = loop.create_task(
        person_follow_loop(get_latest_frame, WS_URL, model, person_follow_stop_event)
    )

    return {"status": "started"}


@app.post("/person_follow/stop")
async def person_follow_stop():
    global person_follow_task, person_follow_stop_event

    if person_follow_stop_event:
        person_follow_stop_event.set()

    try:
        async with websockets.connect(WS_URL) as ws:
            await ws.send("stop,0")
    except:
        pass

    person_follow_task = None
    person_follow_stop_event = None

    return {"status": "stopped"}


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")
