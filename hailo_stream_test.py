#!/usr/bin/env python3
"""
Minimal Hailo object-detection sanity test.

Grabs frames from the ESP32 video stream for a short duration, runs them through
the Hailo HEF, overlays detections using object_detection_post_process, and
writes the annotated video into .output/.
"""

import argparse
import json
import os
import threading
import time
from functools import partial

import cv2
import numpy as np

from common.hailo_inference import HailoInfer
from common.toolbox import default_preprocess, get_labels
from object_detection_post_process import inference_result_handler


DEFAULT_STREAM_URL = "http://192.168.1.50:81/stream"
DEFAULT_HEF = "resources/yolov8s.hef"
DEFAULT_LABELS = "resources/coco_labels.txt"
DEFAULT_CONFIG = "resources/yolo_conf.json"
DEFAULT_OUTPUT_DIR = ".output"


def parse_args():
    parser = argparse.ArgumentParser(description="Quick Hailo stream test (records annotated video).")
    parser.add_argument("--stream-url", default=DEFAULT_STREAM_URL,
                        help="HTTP stream to read frames from (ESP32-CAM).")
    parser.add_argument("--hef", default=DEFAULT_HEF,
                        help="Path to HEF model file.")
    parser.add_argument("--labels", default=DEFAULT_LABELS,
                        help="Path to label file.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Path to YOLO config JSON (visualization params).")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Recording duration in seconds.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write the annotated video into.")
    parser.add_argument("--output-name", default="hailo_test.mp4",
                        help="Filename for the annotated video.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for Hailo inference (default 1).")
    parser.add_argument("--timeout", type=float, default=3.0,
                        help="Seconds to wait for each inference result.")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_frame(frame_bgr, model_w, model_h):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return default_preprocess(rgb, model_w, model_h)


def inference_callback(completion_info, bindings_list, input_batch, result_holder, done_event):
    try:
        if completion_info.exception:
            result_holder["error"] = completion_info.exception
            return

        binding = bindings_list[0]
        if len(binding._output_names) == 1:
            result = binding.output().get_buffer()
        else:
            result = {
                name: np.expand_dims(binding.output(name).get_buffer(), axis=0)
                for name in binding._output_names
            }
        result_holder["result"] = result
        result_holder["frame"] = input_batch[0]
    finally:
        done_event.set()


def run_test(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)

    labels = get_labels(args.labels)
    config_data = load_config(args.config)

    hailo = HailoInfer(args.hef, batch_size=args.batch_size)
    model_h, model_w, _ = hailo.get_input_shape()

    cap = cv2.VideoCapture(args.stream_url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video stream: {args.stream_url}")

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        raise RuntimeError("Failed to grab first frame from stream.")

    frame_h, frame_w = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video writer: {output_path}")

    start_time = time.time()
    frames_processed = 0
    frame = first_frame

    while time.time() - start_time < args.duration:
        if frame is None:
            break

        processed = preprocess_frame(frame, model_w, model_h)
        result_holder = {}
        done_event = threading.Event()
        callback = partial(
            inference_callback,
            input_batch=[frame.copy()],
            result_holder=result_holder,
            done_event=done_event,
        )
        hailo.run([processed], callback)
        done_event.wait(args.timeout)

        if "error" in result_holder:
            raise RuntimeError(f"Inference error: {result_holder['error']}")
        if "result" not in result_holder:
            raise RuntimeError("Inference timed out without results.")

        annotated = inference_result_handler(
            frame.copy(),
            result_holder["result"],
            labels,
            config_data,
            tracker=None,
        )
        writer.write(annotated)
        frames_processed += 1

        ok, frame = cap.read()
        if not ok:
            break

    cap.release()
    writer.release()
    hailo.close()

    print(f"[TEST] Wrote {frames_processed} frames to {output_path}")


def main():
    args = parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
