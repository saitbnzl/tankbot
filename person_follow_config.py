# person_follow_config.py
# --------------------------------------------------
# ALL TUNABLE HYPERPARAMETERS FOR PERSON FOLLOW MODE
# --------------------------------------------------

config = {
    "CONF_THRESHOLD": 0.50,     # YOLO person confidence
    "IMG_SIZE": 320,            # YOLO image size (higher = better accuracy, slower)
    "CENTER_DEADZONE": 0.15,    # ±15% horizontally is “centered”
    "TURN_SPEED": 40,           # rotation speed (0-100)
    "FORWARD_SPEED": 55,        # forward speed (0-100)
    "SEARCH_TURN_SPEED": 30,    # speed when searching for a person
    "LOST_FRAMES_LIMIT": 15,    # after N frames with no person -> stop
    "TRACK_INTERVAL_SEC": 0.05, # delay between control commands
    "MAX_PERSON_AREA": 0.50,    # stop if person covers >50% of image
    "STOP_ON_TOO_CLOSE": True,  # enable stop when too close
}

# easy accessor
def get_config():
    return config

def update_config(new_values: dict):
    for k, v in new_values.items():
        if k in config:
            config[k] = v
    print("[CONFIG] Updated:", config)
    return config
