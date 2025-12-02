# hailo_yolo_adapter.py

import cv2
import numpy as np
from types import SimpleNamespace
from pathlib import Path
from object_detection_utils import preprocess_image, postprocess_detections  # <- adjust names
from utils import HailoAsyncInference  # <- from Hailo examples

HAILO_MODEL_PATH = Path("/home/saitb/resources/yolov8m.hef")
HAILO_LABELS_PATH = Path("/home/saitb/resources/coco_labels.txt")

_hailo_infer = None
_hailo_input_size = (640, 640)  # adjust if your HEF expects different size
_hailo_labels = None

# If you’ve installed HailoRT’s Python package, imports will look like this:
# from hailo_platform import (
#     VDevice, HEF, HailoSchedulingAlgorithm,
#     ConfigureParams, HailoStreamInterface,
#     InferVStreams, InputVStreams, OutputVStreams,
#     InputVStreamParams, OutputVStreamParams, FormatType,
# )
#
# To keep this adapter generic and not over-opinionated,
# the actual Hailo inference call is left as a clearly
# marked TODO block below.


def _init_hailo():
    global _hailo_infer, _hailo_labels

    if _hailo_infer is not None:
        return

    if not HAILO_MODEL_PATH.exists():
        raise RuntimeError(f"Hailo model not found at {HAILO_MODEL_PATH}")

    if not HAILO_LABELS_PATH.exists():
        raise RuntimeError(f"coco labels file not found at {HAILO_LABELS_PATH}")

    # This class comes from the Hailo example (utils.py)
    _hailo_infer = HailoAsyncInference(
        model_path=str(HAILO_MODEL_PATH),
        labels_path=str(HAILO_LABELS_PATH),
        batch_size=1,
    )

    # if there is a helper to load labels, use that; otherwise read file
    _hailo_labels = [line.strip() for line in open(HAILO_LABELS_PATH, "r", encoding="utf-8")]
    print("[HAILO] Initialized with", HAILO_MODEL_PATH)


class SimpleBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf

class SimpleResult:
    def __init__(self, boxes):
        self.boxes = boxes

class _Boxes:
    """
    Minimal Ultralytics-like boxes wrapper:
    - xyxy : (N, 4) float32
    - cls  : (N,) int32
    - conf : (N,) float32
    """
    def __init__(self, xyxy: np.ndarray, cls: np.ndarray, conf: np.ndarray):
        self.xyxy = xyxy.astype(np.float32)
        self.cls = cls.astype(np.int32)
        self.conf = conf.astype(np.float32)

    def __bool__(self):
        return self.xyxy.size > 0


class _Result:
    """
    Ultralytics-like result:
    - boxes : _Boxes
    - names : dict[int, str]
    - plot(): returns annotated image
    """
    def __init__(self, boxes: _Boxes, img: np.ndarray, names: dict[int, str]):
        self.boxes = boxes
        self.orig_img = img
        self.names = names

    def plot(self):
        img = self.orig_img.copy()
        for (x1, y1, x2, y2), cid, conf in zip(
            self.boxes.xyxy, self.boxes.cls, self.boxes.conf
        ):
            x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
            cv2.rectangle(img, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            label = f"{self.names.get(int(cid), str(cid))} {conf:.2f}"
            cv2.putText(
                img,
                label,
                (x1i, max(0, y1i - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return img


class HailoYoloDetector:
    """
    Thin adapter so the rest of your code can stay exactly the same.

    Usage:
        det = HailoYoloDetector("yolov8n_person.hef", "coco_labels.txt", class_filter=[0])
        results = det(frame, imgsz=320, verbose=False)
        r = results[0]
        r.boxes.xyxy, r.boxes.cls, r.boxes.conf, r.plot()

    You implement the actual Hailo inference in _run_hailo().
    """

    def __init__(
        self,
        hef_path: str,
        labels_path: str,
        class_filter: list[int] | None = None,
    ):
        self.hef_path = hef_path
        self.class_filter = set(class_filter or [])

        # Load labels into a dict like Ultralytics model.names
        # Expect standard COCO labels file, one label per line.
        self.names = self._load_labels(labels_path)

        # TODO: initialize Hailo device + load HEF
        # This is intentionally left high-level so you can match
        # it to your installation / HailoRT version.
        #
        # Typical flow (from Hailo docs / examples):
        #   - create VDevice
        #   - load HEF
        #   - configure input/output vstreams
        #
        # self._vdevice = ...
        # self._infer_model = ...
        # self._in_vstreams = ...
        # self._out_vstreams = ...
        #
        # For now we keep them as placeholders:
        self._hailo_ctx = None  # put what you need here

    @staticmethod
    def _load_labels(path: str) -> dict[int, str]:
        names: dict[int, str] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    label = line.strip()
                    if label:
                        names[idx] = label
        except FileNotFoundError:
            # Fallback to empty; class ids will be printed as numbers
            pass
        return names

    # =============== PUBLIC API (Ultralytics-like) ===============

    def __call__(self, frame: np.ndarray, imgsz: int = 320, verbose: bool = False):
        """
        Mimic Ultralytics YOLO __call__ so you don't have to touch
        the rest of the code.

        Returns: [ _Result ]
        """
        if frame is None or frame.size == 0:
            boxes = _Boxes(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                cls=np.zeros((0,), dtype=np.int32),
                conf=np.zeros((0,), dtype=np.float32),
            )
            return [_Result(boxes, frame, self.names)]

        # Preprocess for Hailo: resize to HEF input size (you’ll probably
        # use the fixed model size here, e.g. 640x640). For now we keep
        # it simple and just resize to imgsz x imgsz.
        h0, w0 = frame.shape[:2]
        resized = cv2.resize(frame, (imgsz, imgsz))

        # Run inference on Hailo
        detections = self._run_hailo(resized)

        # detections should be a list of dicts like:
        #   {
        #       "x1": float, "y1": float, "x2": float, "y2": float,
        #       "class_id": int, "confidence": float
        #   }
        # in resized-image coordinates. We map them back to original-frame
        # coordinates to keep behaviour identical to Ultralytics.
        if not detections:
            boxes = _Boxes(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                cls=np.zeros((0,), dtype=np.int32),
                conf=np.zeros((0,), dtype=np.float32),
            )
            return [_Result(boxes, frame, self.names)]

        scale_x = w0 / float(imgsz)
        scale_y = h0 / float(imgsz)

        xyxy_list = []
        cls_list = []
        conf_list = []

        for det in detections:
            cid = int(det["class_id"])
            if self.class_filter and cid not in self.class_filter:
                continue

            x1 = float(det["x1"]) * scale_x
            y1 = float(det["y1"]) * scale_y
            x2 = float(det["x2"]) * scale_x
            y2 = float(det["y2"]) * scale_y

            xyxy_list.append([x1, y1, x2, y2])
            cls_list.append(cid)
            conf_list.append(float(det["confidence"]))

        if not xyxy_list:
            boxes = _Boxes(
                xyxy=np.zeros((0, 4), dtype=np.float32),
                cls=np.zeros((0,), dtype=np.int32),
                conf=np.zeros((0,), dtype=np.float32),
            )
        else:
            xyxy = np.array(xyxy_list, dtype=np.float32)
            cls = np.array(cls_list, dtype=np.int32)
            conf = np.array(conf_list, dtype=np.float32)
            boxes = _Boxes(xyxy=xyxy, cls=cls, conf=conf)

        return [_Result(boxes, frame, self.names)]

    # =============== INTERNAL: where Hailo magic happens ===============

def _run_hailo(frame):
    """
    frame: BGR numpy array (like you already have)
    returns: [SimpleResult]  (so caller can do results[0].boxes.xyxy etc.)
    """
    _init_hailo()
    global _hailo_infer

    # 1) Convert BGR -> RGB (most Hailo examples expect RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2) Resize / preprocess
    # If your object_detection_utils has a preprocess helper, use that instead:
    #   preprocessed = preprocess_image(rgb, input_size=_hailo_input_size)
    # Here’s a generic version:
    resized = cv2.resize(rgb, _hailo_input_size, interpolation=cv2.INTER_LINEAR)
    preprocessed = np.expand_dims(resized, axis=0)  # batch dimension [1,H,W,C], uint8

    # 3) Run inference on Hailo (synchronous)
    # Look at object_detection.py to see how HailoAsyncInference is used.
    # Most likely something like:
    hailo_outputs = _hailo_infer.infer(preprocessed)  # returns list for each image in batch
    # For batch_size=1:
    hailo_output_for_frame = hailo_outputs[0]

    # 4) Postprocess -> bounding boxes
    # In Hailo example this is usually done by a helper.
    # In object_detection.py, search for something like: "postprocess_output" or "postprocess_detection_results".
    # I’ll assume you have a function that does: raw_tensor-> list of {bbox, class_id, score, label}
    detections = postprocess_detections(
        hailo_output_for_frame,
        input_shape=_hailo_input_size,
        num_classes=80,
        confidence_threshold=0.3,
        label_dictionary=_hailo_labels,
    )

    # detections is expected to be a list of dicts:
    # {
    #   "bbox": [x1, y1, x2, y2],
    #   "class_id": int,
    #   "score": float,
    #   "label": str
    # }

    xyxy_list = []
    cls_list = []
    conf_list = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        xyxy_list.append([x1, y1, x2, y2])
        cls_list.append(det["class_id"])
        conf_list.append(det["score"])

    if not xyxy_list:
        boxes = SimpleBoxes(
            xyxy=np.empty((0, 4), dtype=np.float32),
            cls=np.empty((0,), dtype=np.int64),
            conf=np.empty((0,), dtype=np.float32),
        )
    else:
        boxes = SimpleBoxes(
            xyxy=np.array(xyxy_list, dtype=np.float32),
            cls=np.array(cls_list, dtype=np.int64),
            conf=np.array(conf_list, dtype=np.float32),
        )

    return [SimpleResult(boxes)]