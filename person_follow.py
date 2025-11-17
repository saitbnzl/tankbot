# person_follow.py

import asyncio
import time
import websockets
from ultralytics import YOLO

# ==========================
# CONFIG
# ==========================
CONF_THRESHOLD = 0.33
MIN_FOLLOW_CONF = 0.45

CENTER_DEADZONE = 0.15
TURN_SPEED = 60
FORWARD_SPEED = 70

FOLLOW_LOST_GRACE = 10        # FOLLOW → SCAN only after N missed frames
SCAN_LOST_TIMEOUT = 30        # SCAN → IDLE timeout
SEND_RATE_LIMIT = 0.15        # seconds per command


state = "SCAN"


# ==========================
# PERSON DETECTION
# ==========================
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
            best = (x1 + w/2, y1 + h/2, w, h, float(conf))
    return best


# ==========================
# MAIN FOLLOW LOOP
# ==========================
async def person_follow_loop(get_frame, send_motor_command, model: YOLO, stop_event):
    global state

    print("[FOLLOW] Starting follow loop...")

    last_cmd = None
    last_send_time = 0

    lost_follow_frames = 0
    scan_no_person = 0

    while not stop_event.is_set():
        try:
            # reconnect WS loop not needed because send_motor_command handles sending
            await asyncio.sleep(0.01)

            # MAIN LOOP
            frame = get_frame()
            H, W = frame.shape[:2]

            results = model(frame, imgsz=320, verbose=False)[0]
            best = pick_main_person(results)

            # =====================
            # PERSON NOT FOUND
            # =====================
            if best is None:

                # FOLLOW mode grace
                if state == "FOLLOW":
                    lost_follow_frames += 1
                    if lost_follow_frames < FOLLOW_LOST_GRACE:
                        await asyncio.sleep(0.05)
                        continue

                    print(f"[FOLLOW] Lost person for {lost_follow_frames} frames → SCAN")
                    state = "SCAN"
                    scan_no_person = 0
                    lost_follow_frames = 0
                    continue

                # SCAN mode search
                if state == "SCAN":
                    scan_no_person += 1

                    if scan_no_person > SCAN_LOST_TIMEOUT:
                        print("[FOLLOW] Person lost too long → IDLE")
                        await send_motor_command("stop", 0)
                        last_cmd = ("stop", 0)
                        state = "IDLE"
                        scan_no_person = 0
                        await asyncio.sleep(0.05)
                        continue

                    # scan rotate
                    now = time.time()
                    if last_cmd != ("right", TURN_SPEED) and (now - last_send_time) > SEND_RATE_LIMIT:
                        print("[FOLLOW][SCAN] rotating right…")
                        await send_motor_command("right", TURN_SPEED)
                        last_cmd = ("right", TURN_SPEED)
                        last_send_time = now

                    await asyncio.sleep(0.05)
                    continue

                # IDLE mode
                if state == "IDLE":
                    await asyncio.sleep(0.05)
                    continue

            # =====================
            # PERSON FOUND
            # =====================
            cx, cy, w, h, conf = best

            if conf < MIN_FOLLOW_CONF:
                # weak detection
                lost_follow_frames += 1
                if lost_follow_frames > FOLLOW_LOST_GRACE:
                    print("[FOLLOW] Weak confidence → SCAN")
                    state = "SCAN"
                    scan_no_person = 0
                await asyncio.sleep(0.05)
                continue

            lost_follow_frames = 0
            scan_no_person = 0

            if state in ("SCAN", "IDLE"):
                print(f"[FOLLOW] Person detected → FOLLOW (from {state})")
                state = "FOLLOW"

            # Movement
            mid_x = W / 2
            err = (cx - mid_x) / mid_x

            if abs(err) < CENTER_DEADZONE:
                cmd = ("forward", FORWARD_SPEED)
            else:
                cmd = ("left", TURN_SPEED) if err < 0 else ("right", TURN_SPEED)

            now = time.time()
            if cmd != last_cmd and (now - last_send_time) > SEND_RATE_LIMIT:
                print(f"[FOLLOW][TRACK] {cmd[0]} err={err:.2f} conf={conf:.2f}")
                await send_motor_command(cmd[0], cmd[1])
                last_cmd = cmd
                last_send_time = now

            await asyncio.sleep(0.05)

        except Exception as e:
            print("[FOLLOW] Error:", e)
            await asyncio.sleep(0.5)
