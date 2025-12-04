# hailo_runner.py

import threading
import numpy as np
import cv2

import hailo
from hailo_platform import (
    VDevice,
    HEF,
    ConfigureParams,
    InputVStreams,
    OutputVStreams,
    InputVStreamParams,
    OutputVStreamParams,
)

from object_detection_post_process import extract_detections

# ---------- CONFIG (can be overridden via configure_model) ----------
HEF_PATH = "resources/yolov8s.hef"
LABELS_PATH = "resources/coco_labels.txt"

# ---------- GLOBAL STATE ----------
_init_lock = threading.Lock()
_hailo_inited = False

_vdevice = None
_network_group = None
_input_vstreams = None
_output_vstreams = None
_input_shape = None    # (H, W, C)
_labels = []


def configure_model(hef_path: str | None = None, labels_path: str | None = None):
    """
    Allows external code (Detector) to override HEF and labels paths.
    Resets the initialized state so next _run_hailo() will re-init with new paths.
    """
    global HEF_PATH, LABELS_PATH, _hailo_inited

    if hef_path:
        HEF_PATH = hef_path
    if labels_path:
        LABELS_PATH = labels_path

    # Force re-init next time
    _hailo_inited = False


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

        # ⚠️ If your Hailo example uses a different constructor (e.g. .create()),
        # adapt these lines to match it.
        _input_vstreams = InputVStreams(in_infos, _network_group, in_params)
        _output_vstreams = OutputVStreams(out_infos, _network_group, out_params)

        # 4) Infer input tensor shape (assuming single input, NHWC)
        in_info = list(in_infos.values())[0] if isinstance(in_infos, dict) else in_infos[0]
        _input_shape = (in_info.height, in_info.width, in_info.channels)

        # 5) Load labels
        _labels = _load_labels(LABELS_PATH)

        _hailo_inited = True


def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Convert ESP32 frame (BGR uint8, HxWx3) into what the HEF expects.
    """
    H, W, C = _input_shape
    resized = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.uint8)
    return tensor


def _run_hailo(frame_bgr: np.ndarray, config_data: dict, class_filter=None):
    """
    Main entry point for Hailo inference.

    frame_bgr:   numpy, HxWx3, BGR
    config_data: JSON-like dict with post-processing config (for extract_detections)
    class_filter: optional list of class_ids to keep (e.g. [0] for 'person')

    returns: list of detection dicts:
        {
          "class_id": int,
          "class_name": str,
          "confidence": float,
          "bbox": [x1, y1, x2, y2],
        }
    """
    _init_hailo()

    inp = _preprocess(frame_bgr)

    if inp.ndim == 3:
        inp_batch = np.expand_dims(inp, 0)
    else:
        inp_batch = inp

    # Run inference
    with _input_vstreams, _output_vstreams:
        # write input to first input vstream
        list(_input_vstreams.values())[0].write(inp_batch)

        # read outputs from all output vstreams
        raw_outputs = []
        for _, vs in _output_vstreams.items():
            raw_outputs.append(vs.read())

    # Post-process: turn raw Hailo outputs into detection dict
    det_dict = extract_detections(frame_bgr, raw_outputs, config_data)

    boxes = det_dict["detection_boxes"]
    classes = det_dict["detection_classes"]
    scores = det_dict["detection_scores"]
    num_detections = det_dict["num_detections"]

    results = []
    for i in range(num_detections):
        cid = int(classes[i])
        if class_filter and cid not in class_filter:
            continue

        x1, y1, x2, y2 = boxes[i]
        score = float(scores[i])

        class_name = (
            _labels[cid] if 0 <= cid < len(_labels) else str(cid)
        )

        results.append(
            {
                "class_id": cid,
                "class_name": class_name,
                "confidence": score,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            }
        )

    return results
