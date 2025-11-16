from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from fastapi.staticfiles import StaticFiles
import cv2
import asyncio
import websockets
import requests  # BUNU EKLE

from person_follow import person_follow_loop  # BUNU EKLE

import cv2
import asyncio
import websockets

person_follow_task: asyncio.Task | None = None
person_follow_stop_event: asyncio.Event | None = None

VIDEO_URL = "http://192.168.1.50:81/stream"   # ESP32-CAM stream
WS_URL    = "ws://tankbot.local:81"            # your tankbot WS control

app = FastAPI()
model = YOLO("yolov8n.pt")  # or yolov8n.pt in same folder


# ----- Models for API -----
class DriveRequest(BaseModel):
    cmd: str    # "forward", "backward", "left", "right", "stop"
    speed: int = 70  # 0..100


# ----- Helper: read one frame from ESP32-CAM -----
def grab_frame():
    cap = cv2.VideoCapture(VIDEO_URL)
    if not cap.isOpened():
        raise RuntimeError("Cannot open ESP32-CAM stream")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Failed to read frame from ESP32-CAM")

    # 🔄 Kamera sideways olduğu için burada düzelt:
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return frame



# ----- Helper: send WebSocket command to tankbot -----
async def send_ws_command(cmd: str, speed: int):
    async with websockets.connect(WS_URL) as ws:
        await ws.send(f"{cmd},{speed}")


# =============================
#         API ROUTES
# =============================

@app.get("/detect")
def detect():
    """
    Grabs ONE frame from ESP32-CAM, runs YOLO, returns JSON detections.
    """
    try:
        frame = grab_frame()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Run YOLO
    results = model(frame, imgsz=320, verbose=False)[0]

    detections = []
    boxes = results.boxes

    if boxes is not None:
        for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append({
                "class_id": int(cls_id),
                "class_name": model.names[int(cls_id)],
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            })

    return {
        "count": len(detections),
        "detections": detections,
    }


@app.post("/drive")
async def drive(req: DriveRequest):
    """
    Simple relay: send drive command to tankbot through WebSocket.
    """
    # Basic validation
    cmd = req.cmd.lower()
    if cmd not in ("forward", "backward", "left", "right", "stop", "f", "b", "l", "r", "s"):
        raise HTTPException(status_code=400, detail="Invalid cmd")

    speed = max(0, min(100, req.speed))

    try:
        await send_ws_command(cmd, speed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WS error: {e}")

    return {"status": "ok", "sent": {"cmd": cmd, "speed": speed}}

def annotate_frame(frame):
    """
    Run YOLO on a frame and return an annotated frame.
    """
    results = model(frame, imgsz=320, verbose=False)
    annotated = results[0].plot()  # draws boxes + labels
    return annotated


def mjpeg_generator():
    while True:
        try:
            frame = grab_frame()  # zaten döndürülmüş halde geliyor
        except Exception:
            continue

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

def check_video_stream() -> bool:
    """
    Health check by trying to grab ONE frame from the MJPEG stream.
    If this succeeds, the stream is definitely online.
    """
    try:
        cap = cv2.VideoCapture(VIDEO_URL)
        if not cap.isOpened():
            return False

        ok, frame = cap.read()
        cap.release()

        return ok and frame is not None
    except Exception:
        return False



def check_tankbot() -> bool:
    try:
        r = requests.get(f"http://tankbot.local/ping", timeout=1.2)
        text = r.text.lower()
        return "pong" in text
    except:
        return False

@app.get("/status")
def status():
    video_ok = check_video_stream()
    tankbot_ok = check_tankbot()

    return {
        "video_stream": "online" if video_ok else "offline",
        "tankbot": "online" if tankbot_ok else "offline",
    }


@app.get("/video")
def video():
    """
    Live MJPEG stream of YOLO-annotated frames.
    Open in browser: http://pi5.local:8000/video
    """
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.post("/person_follow/start")
async def person_follow_start():
    global person_follow_task, person_follow_stop_event

    # Zaten çalışıyorsa 400 dön
    if person_follow_task is not None and not person_follow_task.done():
        raise HTTPException(status_code=400, detail="Person follow already running")

    # Yeni stop eventi oluştur
    person_follow_stop_event = asyncio.Event()

    # Background task başlat
    loop = asyncio.get_running_loop()
    person_follow_task = loop.create_task(
        person_follow_loop(VIDEO_URL, WS_URL, model, person_follow_stop_event)
    )

    return {"status": "started"}


@app.post("/person_follow/stop")
async def person_follow_stop():
    global person_follow_task, person_follow_stop_event

    if person_follow_stop_event is not None:
        person_follow_stop_event.set()

    # güvenlik: robotu durdurmaya çalış
    try:
        await send_ws_command("stop", 0)
    except Exception:
        pass

    person_follow_task = None
    person_follow_stop_event = None

    return {"status": "stopped"}


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=FileResponse)
def home():
    return FileResponse("static/index.html")
