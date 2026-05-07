#!/usr/bin/env python3
import argparse
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    VDevice,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
)

try:
    from hailo_platform import FormatType
except Exception:
    FormatType = None


DEFAULT_HEF = "./models/yolov8m_seg.hef"
DEFAULT_VIDEO = "./video_inferenza/video_1.mp4"
DEFAULT_OUTPUT = "output_segmentation.mp4"

INPUT_SIZE = 640
NUM_CLASSES = 80
MASK_DIM = 32
REG_MAX = 16
PERSON_CLASS = 0

SCORE_THR = 0.45
NMS_THR = 0.50
MASK_THR = 0.50

DANGER_W_FRAC = 0.40
DANGER_H_FRAC = 0.35

WARNING_MARGIN_FRAC = 0.20
FEET_MASK_FRAC = 0.20

MIN_BOX_AREA_FRAC = 0.001
MAX_BOX_AREA_FRAC = 0.45
MIN_MASK_AREA_FRAC = 0.0003
MAX_MASK_AREA_FRAC = 0.20
MIN_MASK_BOX_IOU = 0.10

def parse_args():
    p = argparse.ArgumentParser(
        description="Hailo YOLOv8 segmentation inference con danger zone"
    )

    p.add_argument("--hef", default=DEFAULT_HEF, help="Path del file HEF")
    p.add_argument("--video", default=DEFAULT_VIDEO, help="Path del video input")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Path del video output")
    p.add_argument("--input-size", type=int, default=INPUT_SIZE, help="Input size del modello")
    p.add_argument("--score-thr", type=float, default=SCORE_THR, help="Confidence threshold")
    p.add_argument("--nms-thr", type=float, default=NMS_THR, help="NMS IoU threshold")
    p.add_argument("--mask-thr", type=float, default=MASK_THR, help="Mask threshold")
    p.add_argument("--person-class", type=int, default=PERSON_CLASS, help="Indice classe person")
    p.add_argument("--danger-w-frac", type=float, default=DANGER_W_FRAC, help="Larghezza danger zone")
    p.add_argument("--danger-h-frac", type=float, default=DANGER_H_FRAC, help="Altezza danger zone")

    p.add_argument(
        "--warning-margin-frac",
        type=float,
        default=WARNING_MARGIN_FRAC,
        help="Estensione extra della warning zone rispetto alla danger zone",
    )
    p.add_argument(
        "--feet-mask-frac",
        type=float,
        default=FEET_MASK_FRAC,
        help="Percentuale inferiore della mask usata come piedi",
    )

    p.add_argument("--display", action="store_true", help="Mostra preview live")
    p.add_argument("--debug-outputs", action="store_true", help="Stampa output solo al primo frame")
    p.add_argument(
        "--person-score-thr",
        type=float,
        default=0.70,
        help="soglia reale per la classe person",
    )
    p.add_argument(
        "--topk-per-scale",
        type=int,
        default=120,
        help="max candidate per scala prima della NMS",
    )
    return p.parse_args()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def letterbox(image: np.ndarray, new_shape: int = 640, color=(114, 114, 114)):
    h, w = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    new_h, new_w = new_shape

    r = min(new_w / w, new_h / h)
    rw, rh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)

    dw = new_w - rw
    dh = new_h - rh
    left = dw // 2
    right = dw - left
    top = dh // 2
    bottom = dh - top

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    meta = {
        "ratio": r,
        "pad": (left, top),
        "resized_shape": (rh, rw),
        "input_shape": (new_h, new_w),
        "orig_shape": (h, w),
    }
    return padded, meta


def preprocess(frame_bgr: np.ndarray, input_size: int):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp, meta = letterbox(rgb, input_size)
    return np.expand_dims(inp.astype(np.uint8), axis=0), meta


def make_vstream_params(network_group):
    try:
        if FormatType is not None:
            in_params = InputVStreamParams.make_from_network_group(network_group)
            out_params = OutputVStreamParams.make_from_network_group(
                network_group, quantized=False, format_type=FormatType.FLOAT32
            )
        else:
            in_params = InputVStreamParams.make_from_network_group(network_group)
            out_params = OutputVStreamParams.make_from_network_group(network_group)
    except TypeError:
        in_params = InputVStreamParams.make_from_network_group(network_group)
        out_params = OutputVStreamParams.make_from_network_group(network_group)
    return in_params, out_params


def inspect_results(results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    parsed = {}
    for name, value in results.items():
        arr = np.asarray(value)
        parsed[name] = arr
    return parsed


def print_output_shapes(parsed: Dict[str, np.ndarray]):
    for name, arr in sorted(parsed.items()):
        print(f"OUTPUT {name}: shape={arr.shape}, dtype={arr.dtype}")


def clip_boxes_xyxy(boxes: np.ndarray, w: int, h: int):
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


def boxes_intersect(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (ax1 < bx2) and (ax2 > bx1) and (ay1 < by2) and (ay2 > by1)


def mask_intersects_zone(mask: np.ndarray, zone):
    x1, y1, x2, y2 = zone
    roi = mask[y1:y2, x1:x2]
    return roi.size > 0 and bool(np.any(roi))


def compute_danger_zone(w: int, h: int, w_frac: float, h_frac: float):
    dw = int(w * w_frac)
    dh = int(h * h_frac)
    cx = w // 2
    x1 = max(0, cx - dw // 2)
    x2 = min(w, cx + dw // 2)
    y2 = h
    y1 = max(0, h - dh)
    return (x1, y1, x2, y2)


def draw_danger_zone(frame, zone, in_alert):
    x1, y1, x2, y2 = zone
    color = (0, 0, 255) if in_alert else (0, 200, 255)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, dst=frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        "DANGER ZONE",
        (x1 + 8, max(24, y1 + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_alert_banner(frame, text="!! ALERT: PERSONA IN ZONA DI PERICOLO !!"):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, dst=frame)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    cv2.putText(
        frame,
        text,
        ((w - tw) // 2, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_info(frame, lines: List[str]):
    if not lines:
        return
    pad = 6
    line_h = 20
    box_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        box_w = max(box_w, tw)
    box_w += pad * 2
    box_h = pad * 2 + line_h * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    for i, line in enumerate(lines):
        y = pad + line_h * (i + 1) - 5
        cv2.putText(
            frame,
            line,
            (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def map_outputs(parsed: Dict[str, np.ndarray]):
    heads = {}
    proto = None

    for name, arr in parsed.items():
        a = np.asarray(arr)
        if a.ndim != 4 or a.shape[0] != 1:
            continue

        h, w, c = a.shape[1], a.shape[2], a.shape[3]

        if (h, w, c) == (160, 160, 32):
            proto = a[0]
            continue

        if h in (20, 40, 80):
            heads.setdefault(h, {})
            if c == 64:
                heads[h]["box"] = a[0]
            elif c == 80:
                heads[h]["cls"] = a[0]
            elif c == 32:
                heads[h]["mask"] = a[0]

    required = [20, 40, 80]
    for s in required:
        if s not in heads or any(k not in heads[s] for k in ("box", "cls", "mask")):
            raise RuntimeError(f"Output incompleti per scala {s}: trovato {list(heads.get(s, {}).keys())}")

    if proto is None:
        raise RuntimeError("Proto tensor 160x160x32 non trovato")

    return heads, proto


def dfl_decode(box_tensor: np.ndarray) -> np.ndarray:
    # box_tensor: (H, W, 64) = 4 * 16 bins
    h, w, c = box_tensor.shape
    if c != 4 * REG_MAX:
        raise RuntimeError(f"Canali box attesi {4 * REG_MAX}, trovati {c}")

    x = box_tensor.reshape(h, w, 4, REG_MAX)
    x = softmax(x, axis=-1)
    bins = np.arange(REG_MAX, dtype=np.float32)
    dist = np.sum(x * bins, axis=-1)
    return dist  # (H, W, 4) => l,t,r,b in grid units


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter = inter_w * inter_h

    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    union = area1 + area2 - inter + 1e-6
    return inter / union


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float):
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    order = np.argsort(-scores)
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        ious = box_iou_xyxy(boxes[i], boxes[order[1:]])
        remain = np.where(ious <= iou_thr)[0]
        order = order[remain + 1]

    return np.array(keep, dtype=np.int32)


def scale_boxes_to_original(boxes_input: np.ndarray, meta: Dict) -> np.ndarray:
    boxes = boxes_input.copy()
    left, top = meta["pad"]
    r = meta["ratio"]
    orig_h, orig_w = meta["orig_shape"]

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r
    boxes = clip_boxes_xyxy(boxes, orig_w, orig_h)
    return boxes

def cls_probs_from_tensor(cls_t: np.ndarray) -> np.ndarray:
    cls_t = cls_t.astype(np.float32)
    mn = float(cls_t.min())
    mx = float(cls_t.max())
    if mn >= 0.0 and mx <= 1.0:
        return cls_t
    return sigmoid(cls_t)


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def box_iou_single(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    xx1 = max(ax1, bx1)
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2)
    yy2 = min(ay2, by2)

    inter_w = max(0.0, xx2 - xx1)
    inter_h = max(0.0, yy2 - yy1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_id = 1 + int(np.argmax(areas))
    return labels == largest_id


def clean_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = keep_largest_component(m > 0)
    return m.astype(bool)


def decode_scale_person_only(
    box_t: np.ndarray,
    cls_t: np.ndarray,
    mask_t: np.ndarray,
    stride: int,
    person_class: int,
    person_score_thr: float,
    topk_per_scale: int,
):
    h, w, _ = box_t.shape
    dists = dfl_decode(box_t)

    cls_probs = cls_probs_from_tensor(cls_t)
    person_scores = cls_probs[..., person_class]

    ys, xs = np.where(person_scores >= person_score_thr)
    if len(xs) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, MASK_DIM), dtype=np.float32),
        )

    scores = person_scores[ys, xs].astype(np.float32)

    if scores.size > topk_per_scale:
        order = np.argsort(-scores)[:topk_per_scale]
        ys = ys[order]
        xs = xs[order]
        scores = scores[order]

    coeffs = mask_t[ys, xs].astype(np.float32)
    d = dists[ys, xs].astype(np.float32)

    cx = xs.astype(np.float32) + 0.5
    cy = ys.astype(np.float32) + 0.5

    x1 = (cx - d[:, 0]) * stride
    y1 = (cy - d[:, 1]) * stride
    x2 = (cx + d[:, 2]) * stride
    y2 = (cy + d[:, 3]) * stride

    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    return boxes, scores, coeffs


def build_masks_person_only(
    proto_hwc: np.ndarray,
    coeffs: np.ndarray,
    boxes_input: np.ndarray,
    meta: Dict,
    mask_thr: float,
):
    proto = np.transpose(proto_hwc, (2, 0, 1)).astype(np.float32)  # (32,160,160)
    c, mh, mw = proto.shape

    proto_flat = proto.reshape(c, -1)                              # (32, mh*mw)
    mask_logits = coeffs @ proto_flat                              # (N, mh*mw)
    mask_logits = mask_logits.reshape(coeffs.shape[0], mh, mw)
    masks_small = sigmoid(mask_logits)

    input_h, input_w = meta["input_shape"]
    left, top = meta["pad"]
    resized_h, resized_w = meta["resized_shape"]
    orig_h, orig_w = meta["orig_shape"]

    scale_x = mw / float(input_w)
    scale_y = mh / float(input_h)

    out_masks = []
    for i in range(coeffs.shape[0]):
        mask_small = masks_small[i]
        x1, y1, x2, y2 = boxes_input[i]

        px1 = max(0, min(mw - 1, int(np.floor(x1 * scale_x))))
        py1 = max(0, min(mh - 1, int(np.floor(y1 * scale_y))))
        px2 = max(0, min(mw, int(np.ceil(x2 * scale_x))))
        py2 = max(0, min(mh, int(np.ceil(y2 * scale_y))))

        cropped = np.zeros_like(mask_small, dtype=np.float32)
        if px2 > px1 and py2 > py1:
            cropped[py1:py2, px1:px2] = mask_small[py1:py2, px1:px2]

        mask_input = cv2.resize(cropped, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        mask_unpad = mask_input[top:top + resized_h, left:left + resized_w]
        mask_full = cv2.resize(mask_unpad, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask_bin = clean_mask(mask_full > mask_thr)
        out_masks.append(mask_bin)

    return out_masks


def filter_person_instances(detections: List[Dict], frame_shape: Tuple[int, int]) -> List[Dict]:
    h, w = frame_shape
    frame_area = float(h * w)
    filtered = []

    for det in detections:
        mask = det["mask"]
        if mask is None:
            continue

        x1, y1, x2, y2 = det["bbox"]
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        box_area = float(bw * bh)

        if box_area < frame_area * MIN_BOX_AREA_FRAC:
            continue
        if box_area > frame_area * MAX_BOX_AREA_FRAC:
            continue

        mask_area = float(mask.sum())
        if mask_area < frame_area * MIN_MASK_AREA_FRAC:
            continue
        if mask_area > frame_area * MAX_MASK_AREA_FRAC:
            continue

        mask_bbox = bbox_from_mask(mask)
        if mask_bbox is None:
            continue

        iou = box_iou_single(np.array([x1, y1, x2, y2], dtype=np.float32), mask_bbox)
        if iou < MIN_MASK_BOX_IOU:
            continue

        filtered.append(det)

    return filtered


def decode_yolov8_seg_person_only(
    parsed: Dict[str, np.ndarray],
    meta: Dict,
    person_class: int,
    person_score_thr: float,
    nms_thr: float,
    mask_thr: float,
    topk_per_scale: int,
):
    heads, proto = map_outputs(parsed)

    all_boxes = []
    all_scores = []
    all_coeffs = []

    stride_by_h = {80: 8, 40: 16, 20: 32}

    for h, stride in stride_by_h.items():
        box_t = heads[h]["box"]
        cls_t = heads[h]["cls"]
        mask_t = heads[h]["mask"]

        boxes, scores, coeffs = decode_scale_person_only(
            box_t=box_t,
            cls_t=cls_t,
            mask_t=mask_t,
            stride=stride,
            person_class=person_class,
            person_score_thr=person_score_thr,
            topk_per_scale=topk_per_scale,
        )

        if len(boxes) > 0:
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_coeffs.append(coeffs)

    if not all_boxes:
        return []

    boxes_input = np.concatenate(all_boxes, axis=0).astype(np.float32)
    scores = np.concatenate(all_scores, axis=0).astype(np.float32)
    coeffs = np.concatenate(all_coeffs, axis=0).astype(np.float32)

    boxes_input = clip_boxes_xyxy(boxes_input, meta["input_shape"][1], meta["input_shape"][0])

    keep = nms_xyxy(boxes_input, scores, nms_thr)
    if keep.size == 0:
        return []

    boxes_input = boxes_input[keep]
    scores = scores[keep]
    coeffs = coeffs[keep]

    boxes_orig = scale_boxes_to_original(boxes_input, meta)
    masks = build_masks_person_only(proto, coeffs, boxes_input, meta, mask_thr)

    detections = []
    for i in range(len(scores)):
        detections.append({
            "bbox_input": boxes_input[i],
            "bbox": tuple(int(round(v)) for v in boxes_orig[i]),
            "score": float(scores[i]),
            "mask": masks[i],
            "class_id": person_class,
        })

    return filter_person_instances(detections, meta["orig_shape"])

def draw_detections(frame: np.ndarray, detections: List[Dict], warning_zone, danger_zone, feet_frac=0.20):
    warning_active = False
    danger_active = False

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        mask = det["mask"]

        in_danger = False
        in_warning = False

        if mask is not None:
            in_danger = feet_mask_intersects_zone(mask, danger_zone, feet_frac)
            in_warning = feet_mask_intersects_zone(mask, warning_zone, feet_frac)
        else:
            in_danger = boxes_intersect((x1, y1, x2, y2), danger_zone)
            in_warning = boxes_intersect((x1, y1, x2, y2), warning_zone)

        in_warning_only = in_warning and not in_danger

        det["in_danger"] = in_danger
        det["in_warning"] = in_warning_only

        danger_active = danger_active or in_danger
        warning_active = warning_active or in_warning_only

        if in_danger:
            color = (0, 0, 255)      # rosso
            tag = " [DANGER]"
            alpha = 0.45
        elif in_warning_only:
            color = (0, 255, 255)    # giallo
            tag = " [WARNING]"
            alpha = 0.30
        else:
            color = (0, 255, 0)      # verde
            tag = ""
            alpha = 0.18

        if mask is not None:
            overlay = frame.copy()
            overlay[mask] = color
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if in_danger:
                cv2.drawContours(frame, contours, -1, (255, 255, 255), 4)
                cv2.drawContours(frame, contours, -1, color, 2)
            else:
                cv2.drawContours(frame, contours, -1, color, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"person {score:.2f}{tag}"
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            color,
            2 if in_danger else 1,
            cv2.LINE_AA,
        )

    return warning_active, danger_active

def draw_zones(frame, warning_zone, danger_zone, warning_active, danger_active):
    wx1, wy1, wx2, wy2 = warning_zone
    dx1, dy1, dx2, dy2 = danger_zone

    overlay = frame.copy()

    cv2.rectangle(overlay, (wx1, wy1), (wx2, wy2), (0, 255, 255), -1)  # giallo
    cv2.rectangle(overlay, (dx1, dy1), (dx2, dy2), (0, 0, 255), -1)    # rosso

    cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, dst=frame)

    cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (0, 255, 255), 2)
    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)

    cv2.putText(
        frame, "WARNING ZONE",
        (wx1 + 8, max(24, wy1 + 24)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA
    )

    cv2.putText(
        frame, "DANGER ZONE",
        (dx1 + 8, max(50, dy1 + 48)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2, cv2.LINE_AA
    )

def extract_feet_mask(mask: np.ndarray, feet_frac: float = 0.20) -> np.ndarray:
    if mask is None:
        return np.zeros((0, 0), dtype=bool)

    if mask.size == 0:
        return np.zeros_like(mask, dtype=bool)

    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros_like(mask, dtype=bool)

    y_top = ys.min()
    y_bottom = ys.max()
    h = max(1, y_bottom - y_top + 1)

    feet_h = max(1, int(round(h * feet_frac)))
    feet_y1 = max(y_top, y_bottom - feet_h + 1)

    feet_mask = np.zeros_like(mask, dtype=bool)
    feet_mask[feet_y1:y_bottom + 1, :] = mask[feet_y1:y_bottom + 1, :]
    return feet_mask


def feet_mask_intersects_zone(mask: np.ndarray, zone, feet_frac: float = 0.20) -> bool:
    if mask is None or mask.size == 0:
        return False

    feet_mask = extract_feet_mask(mask, feet_frac)
    x1, y1, x2, y2 = zone
    roi = feet_mask[y1:y2, x1:x2]
    return roi.size > 0 and bool(np.any(roi))

def main():
    args = parse_args()

    hef = HEF(args.hef)
    with VDevice() as target:
        network_group = target.configure(hef)[0]
        in_params, out_params = make_vstream_params(network_group)

        in_info = network_group.get_input_vstream_infos()[0]
        out_infos = network_group.get_output_vstream_infos()

        print(f"INPUT {in_info.name}: {in_info.shape}")
        for out_info in out_infos:
            print(f"OUTPUT {out_info.name}: {out_info.shape}")

        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"[!] Impossibile aprire il video: {args.video}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (orig_w, orig_h),
        )
        if not writer.isOpened():
            sys.exit(f"[!] Impossibile aprire il writer: {args.output}")

        danger_zone = compute_danger_zone(
            orig_w,
            orig_h,
            args.danger_w_frac,
            args.danger_h_frac,
        )

        warning_w_frac = min(1.0, args.danger_w_frac + args.warning_margin_frac)
        warning_h_frac = min(1.0, args.danger_h_frac + args.warning_margin_frac)
        warning_zone = compute_danger_zone(
            orig_w,
            orig_h,
            warning_w_frac,
            warning_h_frac,
        )

        frame_idx = 0
        t0 = time.time()
        times = []
        printed_debug = False

        with network_group.activate(), InferVStreams(network_group, in_params, out_params) as vstreams:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                inp, meta = preprocess(frame, args.input_size)

                t_inf = time.time()
                raw_results = vstreams.infer({in_info.name: inp})
                parsed = inspect_results(raw_results)
                times.append(time.time() - t_inf)

                if args.debug_outputs and not printed_debug:
                    print_output_shapes(parsed)
                    printed_debug = True

                display = frame.copy()

                try:
                    detections = decode_yolov8_seg_person_only(
                        parsed=parsed,
                        meta=meta,
                        person_class=args.person_class,
                        person_score_thr=args.person_score_thr,
                        nms_thr=args.nms_thr,
                        mask_thr=args.mask_thr,
                        topk_per_scale=args.topk_per_scale,
                    )
                except Exception as e:
                    draw_zones(
                        display,
                        warning_zone,
                        danger_zone,
                        warning_active=False,
                        danger_active=False,
                    )
                    draw_info(display, [
                        f"frame {frame_idx}/{total_frames}",
                        "errore decoder segmentation",
                        str(e)[:90],
                    ])

                    writer.write(display)

                    if args.display:
                        cv2.imshow("Hailo YOLOv8 Seg", display)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    frame_idx += 1
                    continue

                warning_active = False
                danger_active = False

                if detections:
                    warning_active, danger_active = draw_detections(
                        display,
                        detections,
                        warning_zone,
                        danger_zone,
                        feet_frac=args.feet_mask_frac,
                    )

                draw_zones(
                    display,
                    warning_zone,
                    danger_zone,
                    warning_active,
                    danger_active,
                )

                if danger_active:
                    draw_alert_banner(display, "!! ALERT: PERSONA IN DANGER ZONE !!")
                elif warning_active:
                    draw_alert_banner(display, "!! ATTENZIONE: PERSONA IN WARNING ZONE !!")

                avg_ms = 1000.0 * np.mean(times[-30:]) if times else 0.0
                draw_info(display, [
                    f"frame {frame_idx}/{total_frames}",
                    f"instances: {len(detections)} | seg-first",
                    f"infer: {avg_ms:.1f} ms",
                ])

                writer.write(display)

                if args.display:
                    cv2.imshow("Hailo YOLOv8 Seg", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_idx += 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - t0
        fps_e2e = frame_idx / elapsed if elapsed > 0 else 0.0
        print(f"[i] Done. Frame processati: {frame_idx}, fps end-to-end: {fps_e2e:.2f}")
        print(f"[i] Output salvato in: {args.output}")

if __name__ == "__main__":
    main()