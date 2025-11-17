# person_follow.py

import asyncio
import websockets
import time
from ultralytics import YOLO

# -----------------------------
# CONFIGURATION
# -----------------------------

CONF_THRESHOLD = 0.33          # base YOLO threshold
MIN_FOLLOW_CONF = 0.45         # minimum confidence to count as a FOLLOW-worthy detection

CENTER_DEADZONE = 0.15
TURN_SPEED = 60
FORWARD_SPEED = 70

LOST_FRAMES_LIMIT = 30         # SCAN → IDLE timeout
FOLLOW_LOST_GRACE = 10         # FOLLOW doesn't drop on 1–9 missed frames

SEND_RATE_LIMIT = 0.15         # seconds between motor commands (anti-spam)


# -----------------------------
# GLOBAL STATE
# -----------------------------

state = "SCAN"  # start by looking for a person


# -----------------------------
# PERSON PICKING LOGIC
# -----------------------------

def pick_main_person(results):
    """
    Returns (cx, cy, w, h, conf) for the largest person,
    or None if no valid person found.
    """
    boxes = results.boxes
    if not boxes:
        return None

    best = None
    best_area = 0.0

    for box, cid, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        if int(cid) != 0:             # only person class
            continue
        if float(conf) < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(float, box)
        w, h = x2 - x1, y2 - y1
        area = w * h

        if area > best_area:
            best_area = area
            cx = x1 + w / 2
            cy = y1 + h / 2
            best = (cx, cy, w, h, float(conf))

    return best


# -----------------------------
# MAIN FOLLOW LOOP
# -----------------------------

async def person_follow_loop(get_frame_callback, ws_url, model: YOLO, stop_event):
    global state

    print("[FOLLOW] Starting follow loop...")

    last_cmd = None
    last_send_time = 0

    lost_follow_frames = 0    # count short-term, temporary losses
    scan_no_person = 0        # count long-term no-person in SCAN mode

    while not stop_event.is_set():

        try:
            print("[FOLLOW][WS] Connecting to tankbot...")
            async with websockets.connect(ws_url, ping_timeout=None) as ws:
                print("[FOLLOW][WS] Connected!")

                while not stop_event.is_set():

                    # -----------------------
                    # GET FRAME
                    # -----------------------
                    frame = get_frame_callback()
                    H, W = frame.shape[:2]

                    results = model(frame, imgsz=320, verbose=False)[0]
                    best = pick_main_person(results)

                    # -----------------------
                    # PERSON NOT FOUND
                    # -----------------------
                    if best is None:

                        # FOLLOW → stay FOLLOW until grace runs out
                        if state == "FOLLOW":
                            lost_follow_frames += 1

                            if lost_follow_frames < FOLLOW_LOST_GRACE:
                                # ignore short-term drops
                                await asyncio.sleep(0.05)
                                continue

                            print(f"[FOLLOW] Lost person for {lost_follow_frames} frames → SCAN")
                            state = "SCAN"
                            lost_follow_frames = 0
                            scan_no_person = 0
                            continue

                        # SCAN MODE (searching)
                        if state == "SCAN":
                            scan_no_person += 1

                            if scan_no_person > LOST_FRAMES_LIMIT:
                                print("[FOLLOW] Person lost too long → IDLE")
                                now = time.time()
                                if last_cmd != ("stop", 0) and (now - last_send_time) > SEND_RATE_LIMIT:
                                    await ws.send("stop,0")
                                    last_cmd = ("stop", 0)
                                    last_send_time = now
                                state = "IDLE"
                                scan_no_person = 0
                                await asyncio.sleep(0.05)
                                continue

                            # rotate slowly
                            now = time.time()
                            if last_cmd != ("right", TURN_SPEED) and (now - last_send_time) > SEND_RATE_LIMIT:
                                print("[FOLLOW][SCAN] rotating right...")
                                await ws.send(f"right,{TURN_SPEED}")
                                last_cmd = ("right", TURN_SPEED)
                                last_send_time = now

                            await asyncio.sleep(0.05)
                            continue

                        # IDLE MODE
                        if state == "IDLE":
                            if last_cmd != ("stop", 0):
                                print("[FOLLOW][IDLE] Standing by...")
                                await ws.send("stop,0")
                                last_cmd = ("stop", 0)
                            await asyncio.sleep(0.05)
                            continue

                    # -----------------------
                    # PERSON FOUND
                    # -----------------------
                    cx, cy, w, h, conf = best

                    # Ignore detections too weak for FOLLOW
                    if conf < MIN_FOLLOW_CONF:
                        # treat as "person not found"
                        lost_follow_frames += 1
                        if lost_follow_frames >= FOLLOW_LOST_GRACE:
                            print("[FOLLOW] Confidence too low → SCAN")
                            state = "SCAN"
                            lost_follow_frames = 0
                        await asyncio.sleep(0.05)
                        continue

                    # Valid person found → reset all counters
                    lost_follow_frames = 0
                    scan_no_person = 0

                    # FOLLOW MODE entrance
                    if state in ("SCAN", "IDLE"):
                        print(f"[FOLLOW] Person detected → FOLLOW (from {state})")
                        state = "FOLLOW"

                    # CENTERING LOGIC
                    mid_x = W / 2
                    err = (cx - mid_x) / mid_x  # -1..+1

                    if abs(err) < CENTER_DEADZONE:
                        cmd = ("forward", FORWARD_SPEED)
                    else:
                        cmd = ("left", TURN_SPEED) if err < 0 else ("right", TURN_SPEED)

                    # rate limit motor output
                    now = time.time()
                    if cmd != last_cmd and (now - last_send_time) > SEND_RATE_LIMIT:
                        print(f"[FOLLOW][TRACK] {cmd[0]} err={err:.2f} conf={conf:.2f}")
                        await ws.send(f"{cmd[0]},{cmd[1]}")
                        last_cmd = cmd
                        last_send_time = now

                    await asyncio.sleep(0.05)

        except Exception as e:
            print(f"[FOLLOW][WS] disconnected, retrying: {e}")
            await asyncio.sleep(0.5)
