# detector.py

import numpy as np
from ultralytics import YOLO
from hailo_runner import _run_hailo, configure_model


class Detector:
    def __init__(
        self,
        pt_path: str | None = None,
        hef_path: str | None = None,
        labels_path: str | None = None,
        config_data: dict | None = None,
        class_filter: list[int] | None = None,
        use_hailo: bool = False,
    ):
        self.use_hailo = use_hailo
        self.class_filter = class_filter
        self.config_data = config_data

        if self.use_hailo:
            self.mode = "hailo"
            configure_model(hef_path=hef_path, labels_path=labels_path)
        else:
            self.mode = "pytorch"
            if pt_path is None:
                raise ValueError("pt_path is required when use_hailo=False")
            self.model = YOLO(pt_path)

    def __call__(self, frame_bgr, imgsz=None, verbose=False):
        """
        detections = model(frame_bgr, imgsz=..., verbose=...)

        Her iki modda da aynı yapıda döner:
            {
              "class_id": int,
              "class_name": str,
              "confidence": float,
              "bbox": [x1, y1, x2, y2],
            }
        """
        if self.mode == "hailo":
            if self.config_data is None:
                raise ValueError("config_data must be provided for Hailo mode")
            dets = _run_hailo(
                frame_bgr,
                config_data=self.config_data,
                class_filter=self.class_filter,
            )
            return dets

        # PyTorch YOLO path
        results = self.model(frame_bgr, imgsz=imgsz, verbose=verbose)[0]

        dets = []
        for box in results.boxes:
            cid = int(box.cls[0])
            if self.class_filter and cid not in self.class_filter:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            dets.append(
                {
                    "class_id": cid,
                    "class_name": results.names[cid],
                    "confidence": float(box.conf[0]),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
            )
        return dets
