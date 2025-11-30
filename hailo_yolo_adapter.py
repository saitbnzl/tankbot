# hailo_yolo_adapter.py

import cv2
import numpy as np
from types import SimpleNamespace

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

    def _run_hailo(self, resized_bgr: np.ndarray) -> list[dict]:
        """
        Run a single-frame inference on the Hailo device and return
        a list of detection dicts:

            [
                {
                  "x1": float, "y1": float, "x2": float, "y2": float,
                  "class_id": int, "confidence": float,
                },
                ...
            ]

        THIS IS THE ONLY PLACE YOU NEED TO EDIT TO HOOK IN HAILO.

        You can follow the official Hailo runtime Python object detection example:

        - Hailo-Application-Code-Examples/runtime/python/object_detection
          (see README and object_detection.py there)

        and map its outputs into the format above.
        """
        # ====== TODO: replace this stub with real Hailo inference ======
        # For now, return empty detections so nothing breaks structurally.
        # Once you have your Hailo pipeline working, parse its output
        # tensors here and create the list of dicts described above.
        return []
