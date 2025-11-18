# person_follow.py

import asyncio
import time
import traceback

import cv2
import numpy as np
from ultralytics import YOLO

from person_follow_config import get_config  # <-- NEW

# possible states: "SCAN", "FOLLOW", "IDLE", "BLOCKED"
state = "SCAN"

# SMART SEARCH:
# Kişi en son hangi tarafta görüldü? ("left", "right" veya None)
last_seen_side = None

# ADAPTIVE TURN:
# Farklı zeminlerde (halı / fayans vb.) dönüş hızını kameradan ölçüp ayarlamak için
turn_speed_scale = 1.0      # 1.0 = configteki hız
_prev_calib_frame = None    # küçük gri frame saklamak için

# Adaptif sınırlar (istersen config'e taşıyabiliriz)
MIN_TURN_SCALE = 0.75
MAX_TURN_SCALE = 1.5


# ==========================
# PERSON SELECTION
# ==========================
def pick_main_person(results, frame_area: float, conf_threshold: float):
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

        if conf < conf_threshold:
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
# ADAPTIVE TURN HELPERS
# ==========================
def estimate_global_shift_x(prev_small_gray, curr_small_gray):
    """
    İki küçük gri frame arasındaki ortalama yatay hareketi (dx, piksel) döner.

    Outlier'lar (frame skip, ani blur, vs.) için:
    - Optical flow'tan gelen dx map'inin 10–90 percentile aralığını alıyoruz
      ve sadece o aralıktaki piksel hareketlerinden ortalama hesaplıyoruz.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_small_gray,
        curr_small_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )

    dx = flow[..., 0].astype(np.float32).ravel()
    if dx.size == 0:
        return 0.0

    # Percentile clipping: aşırı uç değerleri kırp
    p10, p90 = np.percentile(dx, [10, 90])
    mask = (dx >= p10) & (dx <= p90)

    # Eğer maske çok az piksel içeriyorsa, fallback olarak tüm dx'in mean'ini kullan
    if mask.sum() < dx.size * 0.1:
        mean_dx = float(dx.mean())
    else:
        mean_dx = float(dx[mask].mean())

    return mean_dx


def adaptive_turn_calibration(last_cmd, small):
    """
    Son komut bir dönüş komutuysa (left/right), ardışık iki küçük frame
    arasındaki yatay kaymaya bakarak turn_speed_scale'i ayarlar.

    Amaç:
    - 1.0 etrafında kal,
    - Çok yavaş dönüyorsa scale'i yavaşça ↑ (MAX_TURN_SCALE'e kadar),
    - Çok hızlı dönüyorsa scale'i daha hızlı ↓ (MIN_TURN_SCALE'e kadar),
    - Abs(yaw_norm) aşırı büyükse (örneğin > 0.5) bunu "glitch" sayıp ignore et.
    """
    global _prev_calib_frame, turn_speed_scale

    if small is None:
        return

    if last_cmd is not None and last_cmd[0] in ("left", "right"):
        if _prev_calib_frame is not None:
            try:
                dx = estimate_global_shift_x(_prev_calib_frame, small)
                # 0..1 arası kabaca "dönüş miktarı"
                width = float(small.shape[1]) if small.shape[1] > 0 else 1.0
                yaw_norm = abs(dx) / width

                # Çok saçma büyük değerler (örneğin frame skip / ciddi glitch) → ignore
                if yaw_norm > 0.3:
                    # print(f"[ADAPTIVE] Ignoring outlier yaw_norm={yaw_norm:.3f}")
                    _prev_calib_frame = small
                    return

                # Bu eşikler tamamen deneysel; sahada ayarlayabilirsin
                slow_threshold = 0.02   # bundan küçükse -> çok yavaş dönüyor
                fast_threshold = 0.05   # bundan büyükse -> çok hızlı dönüyor

                # Scale'i biraz yavaş artır / daha hızlı azalt
                if yaw_norm < slow_threshold:
                    # dönmüyor / çok az dönüyor -> hafifçe hızlandır
                    turn_speed_scale *= 1.03   # +%5
                elif yaw_norm > fast_threshold:
                    # çok hızlı dönüyor -> daha ciddi yavaşlat
                    turn_speed_scale *= 0.93   # -%10
                else:
                    # "iyi" aralıkta: yavaşça 1.0'a doğru geri çek
                    if turn_speed_scale > 1.0:
                        turn_speed_scale *= 0.99   # üstten 1'e yaklaş
                    elif turn_speed_scale < 1.0:
                        turn_speed_scale *= 1.01   # alttan 1'e yaklaş

                # Sınırları uygula
                if turn_speed_scale > MAX_TURN_SCALE:
                    turn_speed_scale = MAX_TURN_SCALE
                if turn_speed_scale < MIN_TURN_SCALE:
                    turn_speed_scale = MIN_TURN_SCALE

                # Debug istersen:
                # print(f"[ADAPTIVE] yaw_norm={yaw_norm:.3f} scale={turn_speed_scale:.2f}")

            except Exception as e:
                print("[ADAPTIVE] calibration error:", e)

    # Son frame'i sakla
    _prev_calib_frame = small


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
    global last_seen_side
    global turn_speed_scale

    print("[FOLLOW] Starting follow loop...")

    last_cmd = None           # (cmd, speed)
    last_send_time = 0.0

    lost_follow_frames = 0
    scan_no_person = 0

    while not stop_event.is_set():
        try:
            await asyncio.sleep(0.01)

            # --- read latest config each iteration (dict is mutated in-place) ---
            cfg = get_config()
            CONF_THRESHOLD          = float(cfg.get("CONF_THRESHOLD", 0.50))
            IMG_SIZE                = int(cfg.get("IMG_SIZE", 320))
            CENTER_DEADZONE         = float(cfg.get("CENTER_DEADZONE", 0.15))
            TURN_SPEED_BASE         = int(cfg.get("TURN_SPEED", 60))
            FORWARD_SPEED           = int(cfg.get("FORWARD_SPEED", 70))
            SEARCH_TURN_SPEED_BASE  = int(cfg.get("SEARCH_TURN_SPEED", 40))
            FOLLOW_LOST_GRACE       = int(cfg.get("LOST_FRAMES_GRACE", 30))
            SCAN_LOST_TIMEOUT       = int(cfg.get("LOST_FRAMES_LIMIT", 500))
            SEND_RATE_LIMIT         = float(cfg.get("TRACK_INTERVAL_SEC", 0.2))
            MAX_PERSON_AREA         = float(cfg.get("MAX_PERSON_AREA", 0.70))
            STOP_ON_TOO_CLOSE       = bool(cfg.get("STOP_ON_TOO_CLOSE", True))

            # MIN_FOLLOW_CONF: ayrı key yoksa CONF_THRESHOLD ile aynı olsun
            MIN_FOLLOW_CONF         = float(cfg.get("MIN_FOLLOW_CONF", CONF_THRESHOLD))

            # ---- ADAPTIVE TURN: scaled speeds ----
            effective_turn_speed        = int(TURN_SPEED_BASE * turn_speed_scale)
            effective_search_turn_speed = int(SEARCH_TURN_SPEED_BASE * turn_speed_scale)

            frame = get_frame()
            H, W = frame.shape[:2]
            frame_area = float(W * H)

            # --- ADAPTIVE TURN: küçük gri frame hazırla ve kalibrasyon yap ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small_w = max(W // 4, 1)
            small_h = max(H // 4, 1)
            small = cv2.resize(gray, (small_w, small_h))

            # Bir önceki komut bir dönüş komutu ise, bu frame ile kalibrasyon yap
            adaptive_turn_calibration(last_cmd, small)

            # YOLO tahmini (konfigüre edilebilir IMG_SIZE ile)
            results = model(frame, imgsz=IMG_SIZE, verbose=False)[0]
            best, max_person_ratio = pick_main_person(results, frame_area, CONF_THRESHOLD)

            # ==========================
            # 1) BLOCKED CHECK (STOP_ON_TOO_CLOSE + MAX_PERSON_AREA)
            # ==========================
            if STOP_ON_TOO_CLOSE and max_person_ratio >= MAX_PERSON_AREA:
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
                    # FOLLOW → SCAN geçişinde last_seen_side olduğu gibi kalır (smart search)
                    await asyncio.sleep(0.05)
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

                    # SMART SEARCH:
                    # Eğer last_seen_side biliniyorsa o tarafa dön;
                    # bilinmiyorsa eski davranış: sağa dön.
                    scan_dir = last_seen_side or "right"

                    if last_cmd != (scan_dir, effective_search_turn_speed) and (now - last_send_time) > SEND_RATE_LIMIT:
                        print(
                            f"[FOLLOW][SCAN] rotating {scan_dir}… "
                            f"(smart search, scale={turn_speed_scale:.2f}, "
                            f"speed={effective_search_turn_speed})"
                        )
                        await send_motor_command(scan_dir, effective_search_turn_speed)
                        last_cmd = (scan_dir, effective_search_turn_speed)
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

            # SMART SEARCH:
            # Kişi merkezin anlamlı şekilde sol/sağında ise, o yönü last_seen_side olarak kaydet.
            if err > CENTER_DEADZONE:
                last_seen_side = "right"
            elif err < -CENTER_DEADZONE:
                last_seen_side = "left"
            # Deadzone içindeyse last_seen_side olduğu gibi kalır.

            if abs(err) < CENTER_DEADZONE:
                cmd = ("forward", FORWARD_SPEED)
            else:
                cmd = ("right", effective_turn_speed) if err > 0 else ("left", effective_turn_speed)

            now = time.time()
            if cmd != last_cmd and (now - last_send_time) > SEND_RATE_LIMIT:
                print(
                    f"[FOLLOW][TRACK] {cmd[0]} err={err:.2f} conf={conf:.2f} "
                    f"last_seen_side={last_seen_side} turn_scale={turn_speed_scale:.2f} "
                    f"speed={cmd[1]}"
                )
                await send_motor_command(cmd[0], cmd[1])
                last_cmd = cmd
                last_send_time = now

            await asyncio.sleep(0.05)

        except Exception as e:
            print("[FOLLOW] Error:", e)
            print("[FOLLOW] Last config:", get_config())
            traceback.print_exc()
            await asyncio.sleep(0.5)
