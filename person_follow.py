# person_follow.py

import asyncio
import time
from ultralytics import YOLO

# ==========================
# CONFIG
# ==========================
CONF_THRESHOLD      = 0.35
MIN_FOLLOW_CONF     = 0.4

CENTER_DEADZONE     = 0.16
TURN_SPEED          = 60
FORWARD_SPEED       = 65

FOLLOW_LOST_GRACE   = 30        # FOLLOW → SCAN only after N missed frames
SCAN_LOST_TIMEOUT   = 300        # SCAN → IDLE timeout
SEND_RATE_LIMIT     = 0.1      # seconds per command

BLOCKED_RATIO_THRESHOLD = 0.75  # frame'in %75'i kaplanırsa BLOCKED

# possible states: "SCAN", "FOLLOW", "IDLE", "BLOCKED"
state = "SCAN"


# ==========================
# PERSON SELECTION
# ==========================
def pick_main_person(results, frame_area: float):
    """
    - En küçük alanlı (muhtemelen en uzaktaki) kişiyi seçer.
    - Aynı zamanda frame içindeki en büyük kişi oranını döner (BLOCKED tespiti için).
      Dönüş:
        best_person: (cx, cy, w, h, conf) veya None
        max_ratio:   0.0..1.0 arası, herhangi bir person bounding box'ının frame oranı
    """
    boxes = results.boxes
    if not boxes:
        return None, 0.0

    best = None
    best_area = None  # smallest area
    max_ratio = 0.0

    for box, cid, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        cid = int(cid)
        conf = float(conf)

        # sadece "person" class
        if cid != 0:
            continue

        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(float, box)
        w, h = x2 - x1, y2 - y1
        area = w * h
        ratio = area / frame_area if frame_area > 0 else 0.0

        # BLOCKED için en büyük kişiyi takip et
        if ratio > max_ratio:
            max_ratio = ratio

        # Takip için en küçük kişiyi seç (muhtemelen en uzaktaki)
        if best is None or area < best_area:
            best_area = area
            best = (x1 + w / 2, y1 + h / 2, w, h, conf)

    return best, max_ratio


# ==========================
# MAIN FOLLOW LOOP
# ==========================
async def person_follow_loop(get_frame, send_motor_command, model: YOLO, stop_event: asyncio.Event):
    """
    get_frame:          () -> np.ndarray (BGR frame)
    send_motor_command: (cmd: str, speed: int) -> awaitable
    model:              YOLO instance
    stop_event:         asyncio.Event to stop loop
    """
    global state

    print("[FOLLOW] Starting follow loop...")

    last_cmd = None           # (cmd, speed)
    last_send_time = 0.0

    lost_follow_frames = 0
    scan_no_person = 0

    while not stop_event.is_set():
        try:
            await asyncio.sleep(0.01)

            frame = get_frame()
            H, W = frame.shape[:2]
            frame_area = float(W * H)

            results = model(frame, imgsz=320, verbose=False)[0]
            best, max_person_ratio = pick_main_person(results, frame_area)

            # ==========================
            # 1) BLOCKED CHECK
            # ==========================
            if max_person_ratio >= BLOCKED_RATIO_THRESHOLD:
                # En az bir kişi frame'in büyük kısmını kaplıyor
                if state != "BLOCKED":
                    print(
                        f"[FOLLOW][BLOCKED] Person too close "
                        f"(ratio={max_person_ratio:.2f}) → BLOCKED"
                    )
                state = "BLOCKED"

                now = time.time()
                if last_cmd != ("stop", 0) and (now - last_send_time) > SEND_RATE_LIMIT:
                    await send_motor_command("stop", 0)
                    last_cmd = ("stop", 0)
                    last_send_time = now

                await asyncio.sleep(0.05)
                continue

            # Buraya geliyorsak frame BLOCKED değil
            if state == "BLOCKED":
                # BLOCKED'dan çıktık; kişi hâlâ görülebilir ya da kaybolmuş olabilir
                if best is not None:
                    print("[FOLLOW] BLOCKED cleared, person visible → FOLLOW")
                    state = "FOLLOW"
                else:
                    print("[FOLLOW] BLOCKED cleared, no person → SCAN")
                    state = "SCAN"
                # devam edip normal mantığa düşeceğiz

            # ==========================
            # 2) PERSON YOKSA
            # ==========================
            if best is None:
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

                    now = time.time()
                    if last_cmd != ("right", TURN_SPEED) and (now - last_send_time) > SEND_RATE_LIMIT:
                        print("[FOLLOW][SCAN] rotating right…")
                        await send_motor_command("right", TURN_SPEED)
                        last_cmd = ("right", TURN_SPEED)
                        last_send_time = now

                    await asyncio.sleep(0.05)
                    continue

                if state == "IDLE":
                    await asyncio.sleep(0.05)
                    continue

                # Eğer buraya düşersek (teorik edge case), biraz bekle
                await asyncio.sleep(0.05)
                continue

            # ==========================
            # 3) PERSON VARSA
            # ==========================
            cx, cy, w, h, conf = best

            if conf < MIN_FOLLOW_CONF:
                # Weak detection
                lost_follow_frames += 1
                if lost_follow_frames > FOLLOW_LOST_GRACE:
                    print("[FOLLOW] Weak confidence → SCAN")
                    state = "SCAN"
                    scan_no_person = 0
                await asyncio.sleep(0.05)
                continue

            # Güçlü bir hedefimiz var:
            lost_follow_frames = 0
            scan_no_person = 0

            if state in ("SCAN", "IDLE"):
                print(f"[FOLLOW] Person detected → FOLLOW (from {state})")
                state = "FOLLOW"

            # Hedefe göre yön belirleme
            mid_x = W / 2.0
            err = (cx - mid_x) / mid_x  # -1..+1

            if abs(err) < CENTER_DEADZONE:
                cmd = ("forward", FORWARD_SPEED)
            else:
                cmd = ("right", TURN_SPEED) if err < 0 else ("left", TURN_SPEED)

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
