# tankbot_brain_server.py

from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO
import requests
import asyncio
import websockets
import cv2

from person_follow import person_follow_loop

# ============================================================
#               GLOBAL CAMERA (NO FREEZE!)
# ============================================================

VIDEO_URL = "http://192.168.1.50:81/stream"  # ESP32-CAM MJPEG
WS_URL    = "ws://tankbot.local:81"          # Motor controller

print("[INIT] Opening global VideoCapture...")
cap = cv2.VideoCapture(VIDEO_URL)

if not cap.isOpened():
    raise RuntimeError("Cannot open ESP32-CAM stream")


# ============================================================
#                  GLOBAL YOLO MODEL
# ============================================================

model = YOLO("yolov8n.pt")  # fast lightweight model


# ============================================================
#                  FASTAPI INITIALIZATION
# ============================================================

app = FastAPI()

person_follow_task: asyncio.Task | None = None
person_follow_stop_event: asyncio.Event | None = None


# ============================================================
#                     DATA MODELS
# ============================================================

class DriveRequest(BaseModel):
    cmd: str    # forward/backward/left/right/stop
    speed: int = 70


# ============================================================
#               CAMERA FRAME GRABBING (GLOBAL CAP)
# ============================================================

def grab_frame():
    """
    Reads ONE frame from the global VideoCapture.
    No new VideoCapture is created.
    """
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read frame from ESP32-CAM")

    # rotate sideways camera
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ============================================================
#                 YOLO ANNOTATION FOR /video
# ============================================================

def annotate_frame(frame):
    results = model(frame, imgsz=320, verbose=False)
    annotated = results[0].plot()
    return annotated


# ============================================================
#                   MJPEG STREAM GENERATOR
# ============================================================

def mjpeg_generator():
    while True:
        try:
            frame = grab_frame()
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


# ============================================================
#                   HELPER: SEND WS COMMAND
# ============================================================

async def send_ws_command(cmd: str, speed: int):
    async with websockets.connect(WS_URL) as ws:
        await ws.send(f"{cmd},{speed}")


# ============================================================
#                        ROUTES
# ============================================================

@app.get("/detect")
def detect():
    """
    Read one frame → run YOLO → return detections
    """
    try:
        frame = grab_frame()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = model(frame, imgsz=320, verbose=False)[0]
    detections = []

    if results.boxes is not None:
        for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append({
                "class_id": int(cls_id),
                "class_name": model.names[int(cls_id)],
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
            })

    return {
        "count": len(detections),
        "detections": detections,
    }


@app.post("/drive")
async def drive(req: DriveRequest):
    cmd = req.cmd.lower()
    if cmd not in ("forward", "backward", "left", "right", "stop", "f", "b", "l", "r", "s"):
        raise HTTPException(status_code=400, detail="Invalid cmd")

    speed = max(0, min(100, req.speed))

    try:
        await send_ws_command(cmd, speed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WS error: {e}")

    return {"status": "ok"}


@app.get("/video")
def video():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/status")
def status():
    return {
        "video_stream": "online" if cap.isOpened() else "offline"
    }


@app.post("/person_follow/start")
async def person_follow_start():
    global person_follow_task, person_follow_stop_event

    if person_follow_task is not None and not person_follow_task.done():
        raise HTTPException(status_code=400, detail="Already running")

    person_follow_stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    person_follow_task = loop.create_task(
        person_follow_loop(cap, WS_URL, model, person_follow_stop_event)
    )

    return {"status": "started"}


@app.post("/person_follow/stop")
async def person_follow_stop():
    global person_follow_task, person_follow_stop_event

    if person_follow_stop_event:
        person_follow_stop_event.set()

    try:
        await send_ws_command("stop", 0)
    except:
        pass

    person_follow_task = None
    person_follow_stop_event = None

    return {"status": "stopped"}


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
def home():
    return FileResponse("static/index.html")
