# person_follow.py
import asyncio
import cv2
from ultralytics import YOLO
import websockets

CONF_THRESHOLD = 0.33
CENTER_DEADZONE = 0.15   # ±15% of width
TURN_SPEED = 60
FORWARD_SPEED = 70
LOST_FRAMES_LIMIT = 30   # kaç frame kişi yoksa dur
ROTATE_90_CCW = True     # istersen grab_frame içinde de döndürebilirsin

state = "IDLE"


def pick_main_person(results):
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    best_area = 0.0

    for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        if int(cls_id) != 0:  # COCO: 0 = person
            continue
        if float(conf) < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = [float(v) for v in box]
        w = x2 - x1
        h = y2 - y1
        area = w * h
        if area > best_area:
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            best_area = area
            best = (cx, cy, w, h, float(conf))

    return best


async def person_follow_loop(video_url: str, ws_url: str, model: YOLO, stop_event: asyncio.Event):
    """
    Basit person-follow loop’u.
    - model: global YOLO model (tankbot_brain_server.py'den verilecek)
    - stop_event: /person_follow/stop çağrılınca set edilecek
    """
    global state

    print("[FOLLOW] Starting person follow loop")
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print("[FOLLOW][ERROR] Cannot open video stream")
        return

    try:
        async with websockets.connect(ws_url) as ws:
            last_cmd = None
            no_person_frames = 0

            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[FOLLOW] Failed to read frame")
                    await asyncio.sleep(0.05)
                    continue

                H, W = frame.shape[:2]
                results = model(frame, imgsz=320, verbose=False)[0]

                target = pick_main_person(results)

                # ------------------ PERSON YOK ------------------
                if target is None:
                    # FOLLOW kaybetti → SCAN
                    if state == "FOLLOW":
                        state = "SCAN"

                    # IDLE → SCAN (sen istiyorsun)
                    if state == "IDLE":
                        state = "SCAN"

                    no_person_frames += 1

                    if no_person_frames > LOST_FRAMES_LIMIT:
                        print("[FOLLOW] Person lost too long. Going IDLE.")
                        await ws.send("stop,0")
                        last_cmd = ("stop", 0)
                        state = "IDLE"
                        await asyncio.sleep(0.05)
                        continue

                    # SCAN modunda sağa dön
                    if state == "SCAN" and last_cmd != ("right", TURN_SPEED):
                        print("[FOLLOW][SCAN] Turning right to search")
                        await ws.send(f"right,{TURN_SPEED}")
                        last_cmd = ("right", TURN_SPEED)

                    await asyncio.sleep(0.05)
                    continue

                # ------------------ PERSON VAR ------------------
                if state in ("IDLE", "SCAN"):
                    state = "FOLLOW"

                no_person_frames = 0
                cx, cy, w, h, conf = target
                mid_x = W / 2.0
                err_x = (cx - mid_x) / mid_x  # -1..+1

                if abs(err_x) < CENTER_DEADZONE:
                    cmd = ("forward", FORWARD_SPEED)
                    state_str = "FORWARD"
                else:
                    if err_x < 0:
                        cmd = ("left", TURN_SPEED)
                        state_str = "TURN LEFT"
                    else:
                        cmd = ("right", TURN_SPEED)
                        state_str = "TURN RIGHT"

                if cmd != last_cmd:
                    print(
                        f"[FOLLOW][TRACK] {state_str} | err_x={err_x:.2f} "
                        f"cx={cx:.0f}/{W} area={w*h:.0f} conf={conf:.2f}"
                    )
                    await ws.send(f"{cmd[0]},{cmd[1]}")
                    last_cmd = cmd

                await asyncio.sleep(0.05)

            # güvenlik
            print("[FOLLOW] Loop ended, sending final stop")
            await ws.send("stop,0")

    except Exception as e:
        print(f"[FOLLOW][ERROR] {e}")
