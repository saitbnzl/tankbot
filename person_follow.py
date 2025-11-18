# person_follow.py
import asyncio
import cv2
from ultralytics import YOLO
import websockets

from person_follow_config import get_config

def pick_main_person(results, CONF_THRESHOLD):
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    best_area = 0.0

    for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        if int(cls_id) != 0:
            continue
        if float(conf) < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = [float(v) for v in box]
        w = x2 - x1
        h = y2 - y1
        area = w * h
        if area > best_area:
            cx = x1 + w/2
            cy = y1 + h/2
            best_area = area
            best = (cx, cy, w, h, float(conf))

    return best


async def person_follow_loop(video_url, ws_url, model, stop_event):
    cfg = get_config()
    print("[FOLLOW] LOOP STARTED with config:", cfg, flush=True)

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print("[FOLLOW][ERROR] Cannot open video stream", flush=True)
        return

    try:
        async with websockets.connect(ws_url) as ws:
            last_cmd = None
            no_person_frames = 0

            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    await asyncio.sleep(cfg["TRACK_INTERVAL_SEC"])
                    continue

                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                H, W = frame.shape[:2]
                results = model(frame, imgsz=320, verbose=False)[0]

                target = pick_main_person(results, cfg["CONF_THRESHOLD"])

                if target is None:
                    no_person_frames += 1

                    if no_person_frames > cfg["LOST_FRAMES_LIMIT"]:
                        print("[FOLLOW] LOST PERSON -> STOPPING", flush=True)
                        await ws.send("stop,0")
                        break

                    if last_cmd != ("right", cfg["SEARCH_TURN_SPEED"]):
                        print("[FOLLOW] SEARCHING (turn right)", flush=True)
                        await ws.send(f"right,{cfg['SEARCH_TURN_SPEED']}")
                        last_cmd = ("right", cfg["SEARCH_TURN_SPEED"])

                    await asyncio.sleep(cfg["TRACK_INTERVAL_SEC"])
                    continue

                # person detected
                no_person_frames = 0
                cx, cy, w, h, conf = target
                mid_x = W/2
                err_x = (cx - mid_x) / mid_x

                # too close check
                if cfg["STOP_ON_TOO_CLOSE"]:
                    area_frac = (w*h) / (W*H)
                    if area_frac > cfg["MAX_PERSON_AREA"]:
                        print("[FOLLOW] PERSON TOO CLOSE -> STOP", flush=True)
                        await ws.send("stop,0")
                        break

                # centering logic
                if abs(err_x) < cfg["CENTER_DEADZONE"]:
                    cmd = ("forward", cfg["FORWARD_SPEED"])
                else:
                    if err_x < 0:
                        cmd = ("left", cfg["TURN_SPEED"])
                    else:
                        cmd = ("right", cfg["TURN_SPEED"])

                if cmd != last_cmd:
                    print(f"[FOLLOW] CMD={cmd} err_x={err_x:.2f}", flush=True)
                    await ws.send(f"{cmd[0]},{cmd[1]}")
                    last_cmd = cmd

                await asyncio.sleep(cfg["TRACK_INTERVAL_SEC"])

            await ws.send("stop,0")

    except Exception as e:
        print("[FOLLOW][ERROR]", e, flush=True)
    finally:
        cap.release()
        print("[FOLLOW] LOOP EXIT", flush=True)
