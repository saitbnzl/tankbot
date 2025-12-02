# hailo_runner.py

import threading
import numpy as np
import cv2

import hailo
from hailo_platform import (
    VDevice,
    HEF,
    ConfigureParams,
    InputVStream,
    OutputVStream,
    InputVStreamParams,
    OutputVStreamParams,
)

# ⚠️ You will import the right things from your example here:
#   open runtime/hailo-8/python/object_detection/object_detection_post_process.py
# and see what the postprocess function/class is called.
from object_detection_post_process import postprocess  # <-- adjust name

# ---------- CONFIG ----------
HEF_PATH = "yolov8m.hef"        # or whatever you downloaded
LABELS_PATH = "coco_labels.txt" # same labels file you used before

# ---------- GLOBAL STATE ----------
_init_lock = threading.Lock()
_hailo_inited = False

_vdevice = None
_network_group = None
_input_vstreams = None
_output_vstreams = None
_input_shape = None    # (H, W, C)
_labels = []


def _load_labels(path: str):
    labels = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(line)
    return labels


def _init_hailo():
    global _hailo_inited, _vdevice, _network_group
    global _input_vstreams, _output_vstreams, _input_shape, _labels

    if _hailo_inited:
        return

    with _init_lock:
        if _hailo_inited:
            return

        # 1) Load HEF
        hef = HEF(HEF_PATH)

        # 2) Open device and configure
        _vdevice = VDevice()
        configure_params = hef.create_configure_params(_vdevice)
        _network_group = _vdevice.configure(hef, configure_params)

        # 3) Create vstreams
        in_infos = hef.get_input_vstream_infos()
        out_infos = hef.get_output_vstream_infos()

        in_params = InputVStreamParams()
        out_params = OutputVStreamParams()

        _input_vstreams = InputVStream(in_infos, _network_group, in_params)
        _output_vstreams = OutputVStream(out_infos, _network_group, out_params)

        # 4) Infer input tensor shape (assuming single input, NHWC)
        in_info = list(in_infos.values())[0] if isinstance(in_infos, dict) else in_infos[0]
        _input_shape = (in_info.height, in_info.width, in_info.channels)

        # 5) Load labels
        _labels = _load_labels(LABELS_PATH)

        _hailo_inited = True


def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Convert ESP32 frame (BGR uint8, HxWx3) into what the HEF expects.

    Adjust this to match the pre-process steps in object_detection.py.
    """
    H, W, C = _input_shape
    # Resize to network size
    resized = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

    # Hailo examples usually use RGB uint8, NHWC
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # If the example normalizes / divides by 255 or subtracts mean,
    # do the SAME here. The simplest case is raw uint8:
    tensor = rgb.astype(np.uint8)

    return tensor


def _postprocess(raw_outputs):
    """
    Wrap Hailo's postprocess function so that we always return
    a list of dicts with:
        {class_id, class_name, confidence, bbox=[x1,y1,x2,y2]}
    in *original* frame coordinates.
    """
    # raw_outputs is a dict or list of numpy arrays depending on your HEF.
    # The object_detection_post_process.py already knows how to convert
    # these to boxes – reuse that.

    # This is pseudo; adapt to the actual API in object_detection_post_process.py
    dets = postprocess(raw_outputs)  # [[x1,y1,x2,y2,score,class_id], ...]

    results = []
    for x1, y1, x2, y2, score, cid in dets:
        cid = int(cid)
        results.append(
            {
                "class_id": cid,
                "class_name": _labels[cid] if 0 <= cid < len(_labels) else str(cid),
                "confidence": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            }
        )
    return results


def _run_hailo(frame_bgr: np.ndarray):
    """
    Main entry point you will call from tankbot_brain_server.py

    frame_bgr: numpy, HxWx3, BGR
    returns: list of detection dicts (same as YOLO PyTorch path)
    """
    _init_hailo()

    inp = _preprocess(frame_bgr)

    # Make sure shape matches input vstream; usually (H,W,C) or (1,H,W,C)
    # Check in example: hailo_infer.write_input(...)
    # If batch dimension is needed, add it:
    if inp.ndim == 3:
        inp_batch = np.expand_dims(inp, 0)
    else:
        inp_batch = inp

    # Run inference
    with _input_vstreams, _output_vstreams:
        # Write input (again, check the example; sometimes they expect dict)
        list(_input_vstreams.values())[0].write(inp_batch)

        # Read outputs
        raw_outputs = {}
        for name, vs in _output_vstreams.items():
            raw_outputs[name] = vs.read()

    return _postprocess(raw_outputs)
