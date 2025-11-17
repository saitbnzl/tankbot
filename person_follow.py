# person_follow.py

import asyncio
import websockets
from ultralytics import YOLO
import cv2

CONF_THRESHOLD = 0.33
CENTER_DEADZONE = 0.15
TURN_SPEED = 60
FORWARD_SPEED = 70
LOST_FRAMES_LIMIT = 30

state = "SCAN"  # Start by searching


def pick_main_person(results):
    boxes = results.boxes
    if not boxes:
        return None

    best = None
    best_area = 0
    for box, cid, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        if int(cid) != 0:
            continue
        if float(conf) < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(float, box)
        w, h = x2 - x1, y2 - y1
        area = w * h
        if area > best_area:
            best_area = area
            cx, cy = x1 + w/2, y1 + h/2
            best = (cx, cy, w, h, float(conf))

    return best


async def person_follow_loop(get_frame_callback, ws_url, model: YOLO, stop_event):
    global state

    print("[FOLLOW] Starting person follow...")
    last_cmd = None
    no_person = 0

    try:
        async with websockets.connect(ws_url) as ws:

            while not stop_event.is_set():

                frame = get_frame_callback()
                H, W = frame.shape[:2]

                results = model(frame, imgsz=320, verbose=False)[0]
                target = pick_main_person(results)

                # -----------------------------------------------
                # NO PERSON
                # -----------------------------------------------
                if target is None:

                    if state == "FOLLOW":
                        print("[FOLLOW] Lost person → SCAN")
                        state = "SCAN"
                        no_person = 0

                    elif state == "SCAN":
                        no_person += 1

                        if no_person > LOST_FRAMES_LIMIT:
                            print("[FOLLOW] Person lost too long → IDLE")
                            await ws.send("stop,0")
                            last_cmd = ("stop", 0)
                            state = "IDLE"
                            no_person = 0
                            await asyncio.sleep(0.05)
                            continue

                        if last_cmd != ("right", TURN_SPEED):
                            print("[FOLLOW][SCAN] Turning right to search")
                            await ws.send(f"right,{TURN_SPEED}")
                            last_cmd = ("right", TURN_SPEED)

                    elif state == "IDLE":
                        if last_cmd != ("stop", 0):
                            print("[FOLLOW][IDLE] Standing by...")
                            await ws.send("stop,0")
                            last_cmd = ("stop", 0)

                    await asyncio.sleep(0.05)
                    continue

                # -----------------------------------------------
                # PERSON FOUND
                # -----------------------------------------------
                if state in ("IDLE", "SCAN"):
                    print(f"[FOLLOW] Person detected → FOLLOW (from {state})")
                    state = "FOLLOW"

                no_person = 0

                cx, cy, w, h, conf = target
                mid = W / 2
                err = (cx - mid) / mid

                if abs(err) < CENTER_DEADZONE:
                    cmd = ("forward", FORWARD_SPEED)
                else:
                    cmd = ("left", TURN_SPEED) if err < 0 else ("right", TURN_SPEED)

                if cmd != last_cmd:
                    print(f"[FOLLOW][TRACK] {cmd[0]} err={err:.2f} conf={conf:.2f}")
                    await ws.send(f"{cmd[0]},{cmd[1]}")
                    last_cmd = cmd

                await asyncio.sleep(0.05)

    except Exception as e:
        print(f"[FOLLOW][ERROR] {e}")
