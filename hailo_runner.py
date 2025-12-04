# hailo_runner.py

import threading
import numpy as np
import cv2

import hailo
from hailo_platform import (
    HEF,
    VDevice,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    InferVStreams,
    HailoStreamInterface,
    FormatType,
)


from object_detection_post_process import extract_detections

# ---------- CONFIG (can be overridden via configure_model) ----------
HEF_PATH = "resources/yolov8s.hef"
LABELS_PATH = "resources/coco_labels.txt"

# ---------- GLOBAL STATE ----------
_input_shape = None    # (H, W, C)
_labels = []

_init_lock = threading.Lock()
_hailo_inited = False

_vdevice = None
_network_group = None
_network_group_params = None

_input_vstream_info = None
_output_vstream_info = None
_input_vstreams_params = None
_output_vstreams_params = None

_input_shape = None
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
    global _hailo_inited, _vdevice, _network_group, _network_group_params
    global _input_vstream_info, _output_vstream_info
    global _input_vstreams_params, _output_vstreams_params
    global _input_shape, _labels

    if _hailo_inited:
        return

    with _init_lock:
        if _hailo_inited:
            return

        # 1) Load HEF
        hef = HEF(HEF_PATH)

        # 2) Open device
        _vdevice = VDevice()

        # 3) Configure from HEF (this is the “official” pattern)
        configure_params = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
        )
        _network_group = _vdevice.configure(hef, configure_params)[0]
        _network_group_params = _network_group.create_params()

        # 4) Stream infos
        _input_vstream_info = hef.get_input_vstream_infos()[0]
        _output_vstream_info = hef.get_output_vstream_infos()[0]
        _input_shape = _input_vstream_info.shape  # (H, W, C)

        # 5) Create vstream params (dicts keyed by stream name)
        _input_vstreams_params = InputVStreamParams.make_from_network_group(
            _network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )
        _output_vstreams_params = OutputVStreamParams.make_from_network_group(
            _network_group,
            quantized=False,   # or True / UINT8 depending on your HEF
            format_type=FormatType.FLOAT32,
        )

        # 6) Labels
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
    Returns a list of detection dicts (used by Detector).
    """
    _init_hailo()

    inp = _preprocess(frame_bgr)
    input_data = {
        _input_vstream_info.name: np.expand_dims(inp, axis=0)  # add batch axis
    }

    # Run inference using InferVStreams (no _input_vstreams/_output_vstreams globals)
    with _network_group.activate(_network_group_params):
        with InferVStreams(
            _network_group,
            _input_vstreams_params,
            _output_vstreams_params,
        ) as infer_pipeline:
            results = infer_pipeline.infer(input_data)

    raw_output = results[_output_vstream_info.name]

    return _postprocess(raw_output, frame_bgr, config_data, class_filter)


def _postprocess(raw_outputs, frame_bgr: np.ndarray, config_data: dict, class_filter=None):
    """
    Convert raw Hailo outputs into a list of detection dicts:
        {
          "class_id": int,
          "class_name": str,
          "confidence": float,   # 0..1
          "bbox": [x1, y1, x2, y2],
        }
    """
    dets = extract_detections(frame_bgr, raw_outputs, config_data)

    boxes = dets["detection_boxes"]
    classes = dets["detection_classes"]
    scores = dets["detection_scores"]
    n = dets["num_detections"]

    results = []
    for i in range(n):
        cid = int(classes[i])
        if class_filter and cid not in class_filter:
            continue

        x1, y1, x2, y2 = boxes[i]
        score = float(scores[i])

        results.append(
            {
                "class_id": cid,
                "class_name": _labels[cid] if 0 <= cid < len(_labels) else str(cid),
                "confidence": score,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            }
        )
    return results
