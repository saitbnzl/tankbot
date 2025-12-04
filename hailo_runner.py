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

# Use our own post-processing helper directly
from object_detection_post_process import inference_result_handler

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

        # ⚠️ NOTE:
        # The lines below still need to follow the new Hailo API exactly.
        # Check the official example for the correct way to create InputVStreams / OutputVStreams.
        # For now we keep the structure; just be aware you must adapt these to:
        #   InputVStreams.create(...), OutputVStreams.create(...)
        # or similar, according to your SDK version.
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
    # Resize to network size
    resized = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

    # Hailo examples usually use RGB uint8, NHWC
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # If the example normalizes / divides by 255 or subtracts mean,
    # do the SAME here. The simplest case is raw uint8:
    tensor = rgb.astype(np.uint8)

    return tensor


def _run_hailo(frame_bgr: np.ndarray, config_data: dict, tracker=None):
    """
    Main entry point you will call from tankbot_brain_server.py

    frame_bgr: numpy, HxWx3, BGR
    config_data: JSON-like dict with post-processing config
    tracker: optional BYTETracker instance

    returns: annotated frame (same shape as frame_bgr), with detections drawn.
    """
    _init_hailo()

    inp = _preprocess(frame_bgr)

    # Make sure shape matches input vstream; usually (H,W,C) or (1,H,W,C)
    if inp.ndim == 3:
        inp_batch = np.expand_dims(inp, 0)
    else:
        inp_batch = inp

    # Run inference
    with _input_vstreams, _output_vstreams:
        # Write input (adapt this line to your vstream API if needed)
        list(_input_vstreams.values())[0].write(inp_batch)

        # Read outputs into a list or dict, depending on how your HEF is structured
        raw_outputs = []
        for _, vs in _output_vstreams.items():
            raw_outputs.append(vs.read())

    # Use your custom post-process + drawing logic directly
    # inference_result_handler(original_frame, infer_results, labels, config_data, tracker=None)
    frame_out = inference_result_handler(
        original_frame=frame_bgr.copy(),
        infer_results=raw_outputs,
        labels=_labels,
        config_data=config_data,
        tracker=tracker,
    )

    return frame_out
