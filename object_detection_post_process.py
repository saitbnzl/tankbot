import os
import cv2
import numpy as np
from common.toolbox import id_to_color

# Model input shape is updated by hailo_runner after HEF init.
_MODEL_INPUT_SHAPE: tuple[int, int] | None = None
_PRINTED_DECODE_INFO = False
_STRUCTURE_DUMPED = False
DEBUG_PP = os.environ.get("HAILO_DEBUG_PP", "0") == "1"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def set_model_input_shape(shape):
    """
    Receive the (H, W, C) shape reported by the HEF input stream.
    We only care about height/width for coordinate scaling.
    """
    global _MODEL_INPUT_SHAPE
    if not shape or len(shape) < 2:
        _MODEL_INPUT_SHAPE = None
        return
    try:
        height = int(shape[0])
        width = int(shape[1])
    except (TypeError, ValueError):
        _MODEL_INPUT_SHAPE = None
        return
    _MODEL_INPUT_SHAPE = (height, width)


def inference_result_handler(original_frame, infer_results, labels, config_data, tracker=None):
    # if infer_results is [output] wrap:
    model_output = infer_results[0]  # <-- unwrap the extra level

    detections = extract_detections(original_frame, model_output, config_data)
    frame_with_detections = draw_detections(detections, original_frame, labels, tracker=tracker)
    return frame_with_detections

def draw_detection(image: np.ndarray, box: list, labels: list, score: float, color: tuple, track=False):
    """
    Draw box and label for one detection.

    Args:
        image (np.ndarray): Image to draw on.
        box (list): Bounding box coordinates.
        labels (list): List of labels (1 or 2 elements).
        score (float): Detection score.
        color (tuple): Color for the bounding box.
        track (bool): Whether to include tracking info.
    """
    ymin, xmin, ymax, xmax = map(int, box)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Compose texts
    top_text = f"{labels[0]}: {score:.1f}%" if not track or len(labels) == 2 else f"{score:.1f}%"
    bottom_text = None

    if track:
        if len(labels) == 2:
            bottom_text = labels[1]
        else:
            bottom_text = labels[0]


    # Set colors
    text_color = (255, 255, 255)  # white
    border_color = (0, 0, 0)      # black

    # Draw top text with black border first
    cv2.putText(image, top_text, (xmin + 4, ymin + 20), font, 0.5, border_color, 2, cv2.LINE_AA)
    cv2.putText(image, top_text, (xmin + 4, ymin + 20), font, 0.5, text_color, 1, cv2.LINE_AA)

    # Draw bottom text if exists
    if bottom_text:
        pos = (xmax - 50, ymax - 6)
        cv2.putText(image, bottom_text, pos, font, 0.5, border_color, 2, cv2.LINE_AA)
        cv2.putText(image, bottom_text, pos, font, 0.5, text_color, 1, cv2.LINE_AA)


def denormalize_and_rm_pad(box: list, size: int, padding_length: int, input_height: int, input_width: int) -> list:
    """
    Denormalize bounding box coordinates and remove padding.

    Args:
        box (list): Normalized bounding box coordinates.
        size (int): Size to scale the coordinates.
        padding_length (int): Length of padding to remove.
        input_height (int): Height of the input image.
        input_width (int): Width of the input image.

    Returns:
        list: Denormalized bounding box coordinates with padding removed.
    """
    for i, x in enumerate(box):
        box[i] = int(x * size)
        if (input_width != size) and (i % 2 != 0):
            box[i] -= padding_length
        if (input_height != size) and (i % 2 == 0):
            box[i] -= padding_length

    return box


def _to_flat_float_vector(det):
    """
    Try to convert an arbitrary nested detection structure into a flat 1D float array.
    Returns None if it can't be sensibly converted.
    """
    # First try the simple path
    try:
        arr = np.asarray(det, dtype=float)
        if arr.ndim == 0:
            return None
        if arr.ndim > 1:
            arr = arr.ravel()
        return arr
    except Exception:
        pass

    # Fallback: manually walk and pick only scalar-like items
    flat = []
    try:
        for x in det:
            try:
                flat.append(float(x))
            except Exception:
                # skip non-scalar entries
                continue
        if not flat:
            return None
        return np.asarray(flat, dtype=float)
    except Exception:
        return None


def extract_detections(image: np.ndarray, detections: list, config_data) -> dict:
    """
    Extract detections from raw model outputs.

    Supports two formats:
      1) Dense YOLO-style tensors shaped (N, 4 + 1 + num_classes)
      2) Legacy per-class lists where detections[class_id] holds boxes
    """
    visualization_params = config_data["visualization_params"]
    score_threshold = visualization_params.get("score_thres", 0.5)
    max_boxes = visualization_params.get("max_boxes_to_draw", 50)

    _maybe_dump_structure(detections)

    dense_tensor = _try_get_dense_tensor(detections)
    if dense_tensor is not None:
        decoded = _decode_dense_tensor(image, dense_tensor, score_threshold, max_boxes)
        if decoded is not None:
            return decoded

    return _extract_from_class_lists(
        image,
        detections,
        score_threshold,
        max_boxes,
    )


def _try_get_dense_tensor(detections):
    """
    Attempt to interpret detections as a dense numeric tensor.
    Returns a 2D array with shape (N, C) if possible, otherwise None.
    """
    try:
        arr = np.asarray(detections, dtype=np.float32)
    except Exception:
        return None

    if arr.size == 0:
        return None

    arr = np.squeeze(arr)
    if arr.ndim == 1:
        if arr.size <= 5:
            return None
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        last_dim = arr.shape[-1]
        if last_dim <= 5:
            return None
        try:
            arr = arr.reshape(-1, last_dim)
        except ValueError:
            return None

    if arr.ndim != 2 or arr.shape[1] <= 5:
        return None

    return arr


def _decode_dense_tensor(image, tensor, score_threshold, max_boxes):
    """
    Decode YOLO-style tensors into detection dicts.
    Expected layout per entry: [x, y, w, h, obj, class_probs...]
    """
    num_entries, num_fields = tensor.shape
    if num_fields <= 5:
        return None

    img_height, img_width = image.shape[:2]
    xywh = tensor[:, :4]
    raw_objectness = tensor[:, 4].reshape(num_entries, 1)
    class_scores = tensor[:, 5:]

    if class_scores.size == 0:
        return None

    global _PRINTED_DECODE_INFO
    if not _PRINTED_DECODE_INFO:
        try:
            print(
                f"[PP][DEBUG] Dense tensor shape={tensor.shape}, "
                f"raw obj range=({float(raw_objectness.min()):.3f}, {float(raw_objectness.max()):.3f}), "
                f"class range=({float(class_scores.min()):.3f}, {float(class_scores.max()):.3f})",
                flush=True,
            )
        except Exception:
            pass
        _PRINTED_DECODE_INFO = True

    objectness = _sigmoid(raw_objectness)
    class_scores = _sigmoid(class_scores)
    combined_scores = class_scores * objectness
    class_ids = np.argmax(combined_scores, axis=1)
    scores = combined_scores[np.arange(combined_scores.shape[0]), class_ids]

    keep = scores >= score_threshold
    if not np.any(keep):
        print("[PP] got 0 detections after filtering")
        return {
            "detection_boxes": [],
            "detection_classes": [],
            "detection_scores": [],
            "num_detections": 0,
        }

    xywh = xywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    boxes_xyxy = _xywh_to_xyxy(xywh, img_width, img_height)
    if boxes_xyxy.size == 0:
        print("[PP] got 0 detections after filtering")
        return {
            "detection_boxes": [],
            "detection_classes": [],
            "detection_scores": [],
            "num_detections": 0,
        }

    # Filter invalid boxes (non-positive width/height)
    widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    valid = (widths > 1) & (heights > 1)
    boxes_xyxy = boxes_xyxy[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    if boxes_xyxy.size == 0:
        print("[PP] got 0 detections after filtering")
        return {
            "detection_boxes": [],
            "detection_classes": [],
            "detection_scores": [],
            "num_detections": 0,
        }

    order = np.argsort(scores)[::-1]
    if max_boxes:
        order = order[:max_boxes]
    boxes_xyxy = boxes_xyxy[order]
    scores = scores[order]
    class_ids = class_ids[order]

    print(f"[PP] got {len(scores)} raw detections (dense tensor)")

    return {
        "detection_boxes": boxes_xyxy.tolist(),
        "detection_classes": class_ids.tolist(),
        "detection_scores": scores.tolist(),
        "num_detections": len(scores),
    }


def _xywh_to_xyxy(xywh, img_width, img_height):
    if xywh.size == 0:
        return np.empty((0, 4), dtype=float)

    normalized = float(np.max(np.abs(xywh))) <= 2.0
    if normalized:
        xs = xywh[:, 0] * img_width
        ys = xywh[:, 1] * img_height
        ws = xywh[:, 2] * img_width
        hs = xywh[:, 3] * img_height
    else:
        if _MODEL_INPUT_SHAPE:
            model_h, model_w = _MODEL_INPUT_SHAPE
            scale_x = img_width / float(model_w if model_w else 1)
            scale_y = img_height / float(model_h if model_h else 1)
        else:
            scale_x = scale_y = 1.0
        xs = xywh[:, 0] * scale_x
        ys = xywh[:, 1] * scale_y
        ws = xywh[:, 2] * scale_x
        hs = xywh[:, 3] * scale_y

    x1 = xs - ws / 2.0
    y1 = ys - hs / 2.0
    x2 = xs + ws / 2.0
    y2 = ys + hs / 2.0

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, img_width - 1)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img_height - 1)
    return boxes


def _extract_from_class_lists(image, detections, score_threshold, max_boxes):
    """
    Original per-class extraction logic. Kept for compatibility in case
    the HEF already performs NMS and emits class-separated detections.
    """
    img_height, img_width = image.shape[:2]
    size = max(img_height, img_width)
    padding_length = int(abs(img_height - img_width) / 2)

    all_detections = []
    detections = np.asarray(detections, dtype=object)

    for class_id, detection in enumerate(detections):
        detection = np.asarray(detection, dtype=object)
        if detection.size == 0:
            continue

        if detection.ndim == 1:
            detection = detection.reshape(1, -1)

        for det in detection:
            parsed = _parse_detection_dict(det, class_id, img_height, img_width)
            if parsed is not None:
                score, cid, denorm_bbox = parsed
                if score < score_threshold:
                    continue
            else:
                arr = _to_flat_float_vector(det)
                if arr is None:
                    if DEBUG_PP:
                        print("[PP][DEBUG] skipping non-numeric det:", det)
                    continue

                if arr.size < 5:
                    if DEBUG_PP:
                        print(f"[PP][DEBUG] skipping short det size={arr.size}: {arr}")
                    continue

                bbox = arr[:4]
                score_vec = arr[4:]
                if score_vec.size == 0:
                    continue
                score = float(score_vec[0])
                cid = class_id

                if score < score_threshold:
                    continue

                bbox_list = list(bbox)
                normalized = max(abs(v) for v in bbox_list) <= 1.5
                if normalized:
                    denorm_bbox = denormalize_and_rm_pad(
                        bbox_list,
                        size,
                        padding_length,
                        img_height,
                        img_width,
                    )
                else:
                    x1, y1, x2, y2 = bbox_list[:4]
                    denorm_bbox = [
                        int(np.clip(x1, 0, img_width - 1)),
                        int(np.clip(y1, 0, img_height - 1)),
                        int(np.clip(x2, 0, img_width - 1)),
                        int(np.clip(y2, 0, img_height - 1)),
                    ]

            all_detections.append((score, cid, denorm_bbox))

    all_detections.sort(reverse=True, key=lambda x: x[0])
    top_detections = all_detections[:max_boxes]

    if top_detections:
        scores, class_ids, boxes = zip(*top_detections)
    else:
        scores, class_ids, boxes = [], [], []

    if all_detections:
        print(f"[PP] got {len(all_detections)} raw detections")
    else:
        print("[PP] got 0 detections after filtering")

    return {
        "detection_boxes": list(boxes),
        "detection_classes": list(class_ids),
        "detection_scores": list(scores),
        "num_detections": len(top_detections),
    }



def draw_detections(detections: dict, img_out: np.ndarray, labels, tracker=None):
    """
    Draw detections or tracking results on the image.

    Args:
        detections (dict): Raw detection outputs.
        img_out (np.ndarray): Image to draw on.
        labels (list): List of class labels.
        enable_tracking (bool): Whether to use tracker output (ByteTrack).
        tracker (BYTETracker, optional): ByteTrack tracker instance.

    Returns:
        np.ndarray: Annotated image.
    """

    #extract detection data from the dictionary
    boxes = detections["detection_boxes"]  # List of [xmin,ymin,xmaxm, ymax] boxes
    scores = detections["detection_scores"]  # List of detection confidences
    num_detections = detections["num_detections"]  # Total number of valid detections
    classes = detections["detection_classes"]  # List of class indices per detection

    if tracker:
        dets_for_tracker = []

        #Convert detection format to [xmin,ymin,xmaxm ymax,score] for tracker
        for idx in range(num_detections):
            box = boxes[idx]  #[x, y, w, h]
            score = scores[idx]
            dets_for_tracker.append([*box, score])

        #skip tracking if no detections passed
        if not dets_for_tracker:
            return img_out

        #run BYTETracker and get active tracks
        online_targets = tracker.update(np.array(dets_for_tracker))

        #draw tracked bounding boxes with ID labels
        for track in online_targets:
            track_id = track.track_id  #unique tracker ID
            x1, y1, x2, y2 = track.tlbr  #bounding box (top-left, bottom-right)
            xmin, ymin, xmax, ymax = map(int, [x1, y1, x2, y2])
            best_idx = find_best_matching_detection_index(track.tlbr, boxes)
            if best_idx is None:
                color = (0, 255, 0)  # or some ID-based color
                draw_detection(img_out, [xmin, ymin, xmax, ymax],
                            [f"ID {track_id}"],
                            track.score * 100.0,
                            color,
                            track=True)
            else:
                color = tuple(id_to_color(classes[best_idx]).tolist())
                draw_detection(img_out, [xmin, ymin, xmax, ymax],
                            [labels[classes[best_idx]], f"ID {track_id}"],
                            track.score * 100.0,
                            color,
                            track=True)




    else:
        #No tracking — draw raw model detections
        for idx in range(num_detections):
            color = tuple(id_to_color(classes[idx]).tolist())  #color based on class
            draw_detection(img_out, boxes[idx], [labels[classes[idx]]], scores[idx] * 100.0, color)

    return img_out


def find_best_matching_detection_index(track_box, detection_boxes):
    """
    Finds the index of the detection box with the highest IoU relative to the given tracking box.

    Args:
        track_box (list or tuple): The tracking box in [x_min, y_min, x_max, y_max] format.
        detection_boxes (list): List of detection boxes in [x_min, y_min, x_max, y_max] format.

    Returns:
        int or None: Index of the best matching detection, or None if no match is found.
    """
    best_iou = 0
    best_idx = -1

    for i, det_box in enumerate(detection_boxes):
        iou = compute_iou(track_box, det_box)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    return best_idx if best_idx != -1 else None


def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    IoU measures the overlap between two boxes:
        IoU = (area of intersection) / (area of union)
    Values range from 0 (no overlap) to 1 (perfect overlap).

    Args:
        boxA (list or tuple): [x_min, y_min, x_max, y_max]
        boxB (list or tuple): [x_min, y_min, x_max, y_max]

    Returns:
        float: IoU value between 0 and 1.
    """
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / (areaA + areaB - inter + 1e-5)


def _maybe_dump_structure(detections):
    global _STRUCTURE_DUMPED
    if _STRUCTURE_DUMPED:
        return
    _STRUCTURE_DUMPED = True
    try:
        arr = np.asarray(detections)
        desc = (
            f"[PP][INSPECT] raw type={type(detections).__name__}, "
            f"array_shape={arr.shape}, dtype={arr.dtype}"
        )
        if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
            desc += f", min={float(arr.min()):.4f}, max={float(arr.max()):.4f}"
        print(desc, flush=True)
        if arr.size > 0:
            first = arr.flatten()[0]
            print(f"[PP][INSPECT] first value sample={first}", flush=True)
        return
    except Exception:
        pass
    print(f"[PP][INSPECT] raw detections type={type(detections).__name__}", flush=True)


def _parse_detection_dict(det, fallback_class, img_height, img_width):
    if not isinstance(det, dict):
        return None

    score = det.get("score")
    if score is None:
        score = det.get("confidence", det.get("conf"))
    if score is None:
        return None

    try:
        score = float(score)
    except Exception:
        return None

    cid = det.get("class_id", det.get("class", det.get("cls", fallback_class)))
    try:
        cid = int(cid)
    except Exception:
        cid = fallback_class

    bbox = det.get("bbox")
    if bbox is None:
        keys = ("x1", "y1", "x2", "y2")
        if all(k in det for k in keys):
            bbox = [det[k] for k in keys]
        else:
            keys = ("xmin", "ymin", "xmax", "ymax")
            if all(k in det for k in keys):
                bbox = [det[k] for k in keys]
    if bbox is None:
        return None

    try:
        bbox = [float(v) for v in bbox[:4]]
    except Exception:
        return None

    normalized_flag = det.get("normalized")
    if normalized_flag is True or (
        normalized_flag is None and max(abs(v) for v in bbox) <= 1.5
    ):
        x1 = bbox[0] * img_width
        y1 = bbox[1] * img_height
        x2 = bbox[2] * img_width
        y2 = bbox[3] * img_height
    else:
        x1, y1, x2, y2 = bbox

    denorm_bbox = list(map(int, [x1, y1, x2, y2]))
    return score, cid, denorm_bbox
