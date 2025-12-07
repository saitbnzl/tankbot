# hailo_runner.py

import threading
import traceback
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


from object_detection_post_process import extract_detections, set_model_input_shape
from common.toolbox import default_preprocess

# ---------- CONFIG (can be overridden via configure_model) ----------
HEF_PATH = "resources/yolov8s.hef"
LABELS_PATH = "resources/coco_labels.txt"

# Debug logging flag - set to False to reduce log verbosity during inference
# Can be controlled via environment variable: HAILO_DEBUG_INFERENCE=1
import os
DEBUG_INFERENCE = os.environ.get('HAILO_DEBUG_INFERENCE', '0') == '1'

# ---------- GLOBAL STATE ----------
_input_shape = None    # (H, W, C)
_labels = []

_init_lock = threading.Lock()
_hailo_inited = False

_vdevice = None
_network_group = None
_network_group_params = None

_input_vstream_info = None
_output_vstream_infos = []
_output_vstream_names = []
_input_vstreams_params = None
_output_vstreams_params = None
_raw_output_logged = False



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
    global _hailo_inited, _vdevice, _network_group, _network_group_params
    global _input_vstream_info, _output_vstream_infos, _output_vstream_names
    global _input_vstreams_params, _output_vstreams_params
    global _input_shape, _labels

    if _hailo_inited:
        return

    with _init_lock:
        if _hailo_inited:
            return

        try:
            print(f"[HAILO] Initializing Hailo with HEF: {HEF_PATH}", flush=True)
            
            # 1) Load HEF
            print("[HAILO] Loading HEF file...", flush=True)
            hef = HEF(HEF_PATH)
            print("[HAILO] HEF loaded successfully", flush=True)
            
            # 2) Open device
            print("[HAILO] Opening VDevice...", flush=True)
            _vdevice = VDevice()
            print("[HAILO] VDevice opened successfully", flush=True)
            
            # 3) Configure from HEF (this is the official pattern)
            print("[HAILO] Configuring device with HEF...", flush=True)
            configure_params = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe
            )
            _network_group = _vdevice.configure(hef, configure_params)[0]
            _network_group_params = _network_group.create_params()
            print("[HAILO] Device configured successfully", flush=True)
            
            # 4) Stream infos
            print("[HAILO] Getting stream infos...", flush=True)
            input_infos = hef.get_input_vstream_infos()
            if not input_infos:
                raise RuntimeError("HEF has no input vstreams")
            _input_vstream_info = input_infos[0]
            _input_shape = _input_vstream_info.shape  # (H, W, C)
            set_model_input_shape(_input_shape)
            print(f"[HAILO] Input shape: {_input_shape}", flush=True)

            _output_vstream_infos = hef.get_output_vstream_infos()
            if not _output_vstream_infos:
                raise RuntimeError("HEF has no output vstreams")
            _output_vstream_names = [info.name for info in _output_vstream_infos]
            for info in _output_vstream_infos:
                try:
                    fmt_type = getattr(info.format, "type", None)
                    fmt_order = getattr(info.format, "order", None)
                    print(
                        f"[HAILO] Output stream '{info.name}': shape={info.shape}, "
                        f"format={fmt_type}/{fmt_order}",
                        flush=True,
                    )
                except Exception:
                    print(f"[HAILO] Output stream '{info.name}': shape={info.shape}", flush=True)
            
            # 5) Create vstream params (dicts keyed by stream name)
            print("[HAILO] Creating vstream params...", flush=True)
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
            
            print("[HAILO] Vstream params created successfully", flush=True)
            
            # 6) Labels
            print(f"[HAILO] Loading labels from: {LABELS_PATH}", flush=True)
            _labels = _load_labels(LABELS_PATH)
            print(f"[HAILO] Loaded {len(_labels)} labels", flush=True)
            
            _hailo_inited = True
            print("[HAILO] Initialization complete!", flush=True)
        except Exception as e:
            print(f"[HAILO][ERROR] Initialization failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise



def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Hailo HEF models are exported expecting Ultralytics letterbox preprocessing.
    Use the shared default_preprocess (padding + resize) to preserve aspect ratio.
    """
    if _input_shape is None:
        raise RuntimeError("Hailo input shape not initialized")

    model_h, model_w, _ = _input_shape
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    letterboxed = default_preprocess(rgb, model_w, model_h)
    letterboxed = letterboxed.astype(np.float32) / 255.0
    return letterboxed


def _run_hailo(frame_bgr: np.ndarray, config_data: dict, class_filter=None):
    """
    Main entry point for Hailo inference.
    Returns a list of detection dicts (used by Detector).
    """
    try:
        if DEBUG_INFERENCE:
            print("[HAILO] Starting inference...", flush=True)
        _init_hailo()

        if DEBUG_INFERENCE:
            print("[HAILO] Preprocessing frame...", flush=True)
        inp = _preprocess(frame_bgr)
        input_data = {
            _input_vstream_info.name: np.expand_dims(inp, axis=0)  # add batch axis
        }

        if DEBUG_INFERENCE:
            print("[HAILO] Running inference on device...", flush=True)
        # Run inference using InferVStreams (no _input_vstreams/_output_vstreams globals)
        with _network_group.activate(_network_group_params):
            with InferVStreams(
                _network_group,
                _input_vstreams_params,
                _output_vstreams_params,
            ) as infer_pipeline:
                if DEBUG_INFERENCE:
                    print("[HAILO] Calling infer_pipeline.infer()...", flush=True)
                results = infer_pipeline.infer(input_data)
                if DEBUG_INFERENCE:
                    print("[HAILO] Inference completed", flush=True)

        _log_raw_output_structure(results)
        raw_output = _select_primary_output(results)

        if DEBUG_INFERENCE:
            print("[HAILO] Post-processing results...", flush=True)
        detections = _postprocess(raw_output, frame_bgr, config_data, class_filter)
        if DEBUG_INFERENCE:
            print(f"[HAILO] Inference successful, found {len(detections)} detections", flush=True)
        return detections
    except Exception as e:
        print(f"[HAILO][ERROR] Inference failed: {e}", flush=True)
        traceback.print_exc()
        raise


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


def _log_raw_output_structure(raw_outputs):
    global _raw_output_logged
    if _raw_output_logged:
        return
    _raw_output_logged = True
    try:
        if isinstance(raw_outputs, dict):
            for name, value in raw_outputs.items():
                _print_array_info(f"[HAILO][INSPECT] output[{name}]", value)
        else:
            _print_array_info("[HAILO][INSPECT] raw output", raw_outputs)
    except Exception as exc:
        preview = str(raw_outputs)
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"[HAILO][INSPECT] raw output uninspectable ({exc}): {preview}", flush=True)


def _print_array_info(label, value):
    arr = np.asarray(value)
    desc = f"{label} type={type(value).__name__}, shape={arr.shape}, dtype={arr.dtype}"
    if arr.size:
        desc += f", min={float(arr.min()):.4f}, max={float(arr.max()):.4f}"
    print(desc, flush=True)
    if arr.size:
        sample = arr.flatten()[: min(10, arr.size)]
        print(f"{label} sample={sample}", flush=True)


def _select_primary_output(results):
    if not isinstance(results, dict):
        return results
    # Prefer configured output names
    if _output_vstream_names:
        for name in _output_vstream_names:
            if name in results:
                return results[name]
    # Fallback to first entry
    for value in results.values():
        return value
    raise RuntimeError("Infer results dictionary is empty")
