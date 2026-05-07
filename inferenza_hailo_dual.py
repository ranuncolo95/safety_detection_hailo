#!/usr/bin/env python3
"""
inferenza_hailo_dual.py
-----------------------
Inferenza doppia su Raspberry Pi 5 + Hailo-8L:
    - YOLOv8m-seg  : segmentazione persone
    - SCDepthV3    : depth estimation (calibrata in metri)

La classificazione "DANGER / WARNING / SAFE" è basata sulla depth metrica
ai piedi della persona, non più su una zona pixel fissa.

Soglie default:
    - persona < DANGER_DIST  metri  →  rosso pieno + alert
    - persona < WARNING_DIST metri  →  giallo
    - altrimenti                    →  verde (contorno)

Pipeline per frame:
    1. preprocess YOLO (letterbox 640x640 uint8)
    2. inferenza YOLO su Hailo  → tensori detection + proto mask
    3. decode YOLO → maschere binarie persona (su risoluzione originale)
    4. preprocess SCDepth (256x320 uint8)
    5. inferenza SCDepth su Hailo → log-depth raw
    6. calibrazione  → depth_m (metri reali)
    7. classify_person_by_feet → label DANGER/WARNING/SAFE
    8. overlay + scrittura video

Esempio:
    python inferenza_hailo_dual.py \\
        --hef-yolo models/yolov8m_seg.hef \\
        --hef-depth models/scdepthv3.hef \\
        --calib calib_params.json \\
        --video video_inferenza/video_1.mp4 \\
        --output output_dual.mp4 \\
        --display
"""

import argparse
import os
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

from depth_calibrato import DepthCalibrator


# ============================================================================
# Costanti modelli
# ============================================================================
# YOLO
YOLO_INPUT_SIZE = 640
YOLO_NUM_CLASSES = 80
YOLO_MASK_DIM = 32
YOLO_REG_MAX = 16
PERSON_CLASS = 0

# SCDepthV3 (Hailo Model Zoo: 256x320x3 input)
SCDEPTH_INPUT_H = 256
SCDEPTH_INPUT_W = 320

# Filtri YOLO
SCORE_THR = 0.45
NMS_THR = 0.50
MASK_THR = 0.50
PERSON_SCORE_THR = 0.70
TOPK_PER_SCALE = 120

MIN_BOX_AREA_FRAC = 0.001
MAX_BOX_AREA_FRAC = 0.45
MIN_MASK_AREA_FRAC = 0.0003
MAX_MASK_AREA_FRAC = 0.20
MIN_MASK_BOX_IOU = 0.10

# Soglie depth (metri) per la classificazione
DANGER_DIST_DEFAULT = 3.0
WARNING_DIST_DEFAULT = 5.0

# Visualizzazione overlay prossimità (metri)
PROX_OVERLAY_ALPHA = 0.20

# Classificazione piedi
FOOT_FRAC = 0.20
MIN_VALID_FOOT_PIXELS = 30
CLOSE_RATIO_THRESH = 0.35

DEPTH_CLAMP_MAX = 50.0  # metri, oltre è rumore di calibrazione


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hef-yolo", default="./models/yolov8m_seg.hef")
    p.add_argument("--hef-depth", default="./models/scdepthv3.hef")
    p.add_argument("--calib", default="./calib_params.json",
                   help="JSON parametri calibrazione SCDepthV3 → metri")
    p.add_argument("--video", default="./video_inferenza/video_1.mp4")
    p.add_argument("--output", default="output_hailo_dual.mp4")
    p.add_argument("--display", action="store_true")
    p.add_argument("--debug-outputs", action="store_true",
                   help="Stampa le shape degli output al primo frame")

    # Soglie distanza
    p.add_argument("--danger-dist", type=float, default=DANGER_DIST_DEFAULT,
                   help=f"Distanza per DANGER in metri (default: {DANGER_DIST_DEFAULT})")
    p.add_argument("--warning-dist", type=float, default=WARNING_DIST_DEFAULT,
                   help=f"Distanza per WARNING in metri (default: {WARNING_DIST_DEFAULT})")

    # YOLO
    p.add_argument("--score-thr", type=float, default=SCORE_THR)
    p.add_argument("--nms-thr", type=float, default=NMS_THR)
    p.add_argument("--mask-thr", type=float, default=MASK_THR)
    p.add_argument("--person-score-thr", type=float, default=PERSON_SCORE_THR)
    p.add_argument("--topk-per-scale", type=int, default=TOPK_PER_SCALE)

    return p.parse_args()


# ============================================================================
# Utilities generiche
# ============================================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def make_vstream_params(network_group):
    """Output FLOAT32 quando possibile (necessario per il decoder YOLO)."""
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


# ============================================================================
# YOLO: preprocessing
# ============================================================================
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


def preprocess_yolo(frame_bgr: np.ndarray, input_size: int = YOLO_INPUT_SIZE):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp, meta = letterbox(rgb, input_size)
    return np.expand_dims(inp.astype(np.uint8), axis=0), meta


# ============================================================================
# YOLO: decoder (uguale a quello della tua inferenza.py)
# ============================================================================
def map_outputs_yolo(parsed: Dict[str, np.ndarray]):
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

    for s in (20, 40, 80):
        if s not in heads or any(k not in heads[s] for k in ("box", "cls", "mask")):
            raise RuntimeError(f"Output YOLO incompleti per scala {s}")
    if proto is None:
        raise RuntimeError("Proto tensor 160x160x32 non trovato")
    return heads, proto


def dfl_decode(box_tensor: np.ndarray) -> np.ndarray:
    h, w, c = box_tensor.shape
    if c != 4 * YOLO_REG_MAX:
        raise RuntimeError(f"Canali box attesi {4 * YOLO_REG_MAX}, trovati {c}")
    x = box_tensor.reshape(h, w, 4, YOLO_REG_MAX)
    x = softmax(x, axis=-1)
    bins = np.arange(YOLO_REG_MAX, dtype=np.float32)
    return np.sum(x * bins, axis=-1)  # (H, W, 4)


def cls_probs_from_tensor(cls_t: np.ndarray) -> np.ndarray:
    cls_t = cls_t.astype(np.float32)
    if cls_t.min() >= 0.0 and cls_t.max() <= 1.0:
        return cls_t
    return sigmoid(cls_t)


def clip_boxes_xyxy(boxes: np.ndarray, w: int, h: int):
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
    return boxes


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    a1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    a2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter + 1e-6)


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
        order = order[1:][ious <= iou_thr]
    return np.array(keep, dtype=np.int32)


def scale_boxes_to_original(boxes_input: np.ndarray, meta: Dict) -> np.ndarray:
    boxes = boxes_input.copy()
    left, top = meta["pad"]
    r = meta["ratio"]
    orig_h, orig_w = meta["orig_shape"]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r
    return clip_boxes_xyxy(boxes, orig_w, orig_h)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return labels == (1 + int(np.argmax(areas)))


def clean_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return keep_largest_component(m > 0).astype(bool)


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def box_iou_single(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + ab - inter + 1e-6)


def decode_scale_person(box_t, cls_t, mask_t, stride, person_score_thr, topk):
    h, w, _ = box_t.shape
    dists = dfl_decode(box_t)
    cls_probs = cls_probs_from_tensor(cls_t)
    person_scores = cls_probs[..., PERSON_CLASS]

    ys, xs = np.where(person_scores >= person_score_thr)
    if len(xs) == 0:
        return (np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, YOLO_MASK_DIM), dtype=np.float32))

    scores = person_scores[ys, xs].astype(np.float32)
    if scores.size > topk:
        order = np.argsort(-scores)[:topk]
        ys, xs, scores = ys[order], xs[order], scores[order]

    coeffs = mask_t[ys, xs].astype(np.float32)
    d = dists[ys, xs].astype(np.float32)
    cx = xs.astype(np.float32) + 0.5
    cy = ys.astype(np.float32) + 0.5
    x1 = (cx - d[:, 0]) * stride
    y1 = (cy - d[:, 1]) * stride
    x2 = (cx + d[:, 2]) * stride
    y2 = (cy + d[:, 3]) * stride
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32), scores, coeffs


def build_masks(proto_hwc, coeffs, boxes_input, meta, mask_thr):
    proto = np.transpose(proto_hwc, (2, 0, 1)).astype(np.float32)  # (32, 160, 160)
    c, mh, mw = proto.shape
    mask_logits = (coeffs @ proto.reshape(c, -1)).reshape(coeffs.shape[0], mh, mw)
    masks_small = sigmoid(mask_logits)

    input_h, input_w = meta["input_shape"]
    left, top = meta["pad"]
    rh, rw = meta["resized_shape"]
    orig_h, orig_w = meta["orig_shape"]
    sx, sy = mw / float(input_w), mh / float(input_h)

    out = []
    for i in range(coeffs.shape[0]):
        x1, y1, x2, y2 = boxes_input[i]
        px1 = max(0, min(mw - 1, int(np.floor(x1 * sx))))
        py1 = max(0, min(mh - 1, int(np.floor(y1 * sy))))
        px2 = max(0, min(mw, int(np.ceil(x2 * sx))))
        py2 = max(0, min(mh, int(np.ceil(y2 * sy))))

        cropped = np.zeros_like(masks_small[i], dtype=np.float32)
        if px2 > px1 and py2 > py1:
            cropped[py1:py2, px1:px2] = masks_small[i, py1:py2, px1:px2]

        m_in = cv2.resize(cropped, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        m_unpad = m_in[top:top + rh, left:left + rw]
        m_full = cv2.resize(m_unpad, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        out.append(clean_mask(m_full > mask_thr))
    return out


def filter_person_instances(detections, frame_shape):
    h, w = frame_shape
    area = float(h * w)
    out = []
    for det in detections:
        mask = det["mask"]
        if mask is None:
            continue
        x1, y1, x2, y2 = det["bbox"]
        ba = float(max(0, x2 - x1) * max(0, y2 - y1))
        ma = float(mask.sum())
        if not (area * MIN_BOX_AREA_FRAC <= ba <= area * MAX_BOX_AREA_FRAC):
            continue
        if not (area * MIN_MASK_AREA_FRAC <= ma <= area * MAX_MASK_AREA_FRAC):
            continue
        mb = bbox_from_mask(mask)
        if mb is None or box_iou_single(np.array([x1, y1, x2, y2], dtype=np.float32), mb) < MIN_MASK_BOX_IOU:
            continue
        out.append(det)
    return out


def decode_yolov8_seg(parsed, meta, person_score_thr, nms_thr, mask_thr, topk):
    heads, proto = map_outputs_yolo(parsed)
    all_b, all_s, all_c = [], [], []
    for h, stride in {80: 8, 40: 16, 20: 32}.items():
        b, s, c = decode_scale_person(
            heads[h]["box"], heads[h]["cls"], heads[h]["mask"],
            stride, person_score_thr, topk,
        )
        if len(b) > 0:
            all_b.append(b); all_s.append(s); all_c.append(c)

    if not all_b:
        return []

    boxes = np.concatenate(all_b, axis=0).astype(np.float32)
    scores = np.concatenate(all_s, axis=0).astype(np.float32)
    coeffs = np.concatenate(all_c, axis=0).astype(np.float32)
    boxes = clip_boxes_xyxy(boxes, meta["input_shape"][1], meta["input_shape"][0])

    keep = nms_xyxy(boxes, scores, nms_thr)
    if keep.size == 0:
        return []
    boxes, scores, coeffs = boxes[keep], scores[keep], coeffs[keep]

    boxes_orig = scale_boxes_to_original(boxes, meta)
    masks = build_masks(proto, coeffs, boxes, meta, mask_thr)

    detections = [
        {
            "bbox_input": boxes[i],
            "bbox": tuple(int(round(v)) for v in boxes_orig[i]),
            "score": float(scores[i]),
            "mask": masks[i],
            "class_id": PERSON_CLASS,
        }
        for i in range(len(scores))
    ]
    return filter_person_instances(detections, meta["orig_shape"])


# ============================================================================
# SCDepthV3: pre/post-processing
# ============================================================================
def preprocess_scdepth(frame_bgr: np.ndarray) -> np.ndarray:
    """Hailo SCDepthV3: input 256x320x3 uint8 RGB."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (SCDEPTH_INPUT_W, SCDEPTH_INPUT_H),
                         interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(resized.astype(np.uint8), axis=0)


def postprocess_scdepth(
    raw_results: Dict[str, np.ndarray],
    calibrator: DepthCalibrator,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """raw_results → depth in metri, risoluzione (H_orig, W_orig)."""
    # SCDepth ha tipicamente un solo output, ma cerchiamolo robust-mente
    raw = None
    for name, arr in raw_results.items():
        a = np.asarray(arr)
        # cerca tensore tipo (1, H, W, 1) o (1, H, W) o (1, 1, H, W)
        if a.ndim in (3, 4):
            raw = a
            break
    if raw is None:
        raise RuntimeError("Output SCDepthV3 non riconosciuto")

    # squeeze a 2D (H, W)
    if raw.ndim == 4:
        # (1, H, W, 1) o (1, 1, H, W)
        if raw.shape[0] == 1 and raw.shape[-1] == 1:
            raw = raw[0, :, :, 0]
        elif raw.shape[0] == 1 and raw.shape[1] == 1:
            raw = raw[0, 0, :, :]
        else:
            raw = raw.squeeze()
    elif raw.ndim == 3:
        raw = raw.squeeze()

    raw = raw.astype(np.float32)

    # resize alla risoluzione originale
    H, W = target_shape
    raw_full = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)

    # calibrazione → metri
    depth_m = calibrator.to_meters(raw_full, clamp_min=0.0, clamp_max=DEPTH_CLAMP_MAX)
    return depth_m


# ============================================================================
# Classificazione persona via depth ai piedi
# ============================================================================
def classify_person_by_feet(
    mask: np.ndarray,
    depth_m: np.ndarray,
    danger_dist: float,
    warning_dist: float,
    foot_frac: float = FOOT_FRAC,
    min_valid_pixels: int = MIN_VALID_FOOT_PIXELS,
    close_ratio_thresh: float = CLOSE_RATIO_THRESH,
):
    """
    Returns:
        level: "danger" | "warning" | "safe"
        info: dict con median_depth, close_ratio, valid_pixels, bbox
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return "safe", None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h = y2 - y1 + 1
    foot_h = max(1, int(h * foot_frac))
    y_foot = max(y1, y2 - foot_h + 1)

    foot_mask = np.zeros_like(mask, dtype=bool)
    foot_mask[y_foot:y2 + 1, x1:x2 + 1] = True
    foot_mask &= mask

    foot_depth = depth_m[foot_mask]
    valid = np.isfinite(foot_depth) & (foot_depth > 0.0) & (foot_depth < DEPTH_CLAMP_MAX)
    foot_depth = foot_depth[valid]

    info = {
        "valid_pixels": int(foot_depth.size),
        "bbox": (int(x1), int(y1), int(x2), int(y2)),
    }

    if foot_depth.size < min_valid_pixels:
        info["median_depth"] = None
        info["close_ratio_danger"] = 0.0
        info["close_ratio_warning"] = 0.0
        return "safe", info

    median_depth = float(np.median(foot_depth))
    close_ratio_danger = float(np.mean(foot_depth <= danger_dist))
    close_ratio_warning = float(np.mean(foot_depth <= warning_dist))

    info["median_depth"] = median_depth
    info["close_ratio_danger"] = close_ratio_danger
    info["close_ratio_warning"] = close_ratio_warning

    if (close_ratio_danger >= close_ratio_thresh) or (median_depth <= danger_dist):
        return "danger", info
    if (close_ratio_warning >= close_ratio_thresh) or (median_depth <= warning_dist):
        return "warning", info
    return "safe", info


# ============================================================================
# Visualizzazione
# ============================================================================
def proximity_overlay(frame_bgr, depth_m, max_distance, alpha=PROX_OVERLAY_ALPHA):
    """Heatmap rosso→giallo nelle aree con depth < max_distance."""
    depth_clip = np.clip(depth_m, 0, max_distance)
    depth_norm = (depth_clip / max_distance * 255).astype(np.uint8)
    color_layer = cv2.applyColorMap(depth_norm, cv2.COLORMAP_AUTUMN)
    far = depth_m >= max_distance
    color_layer[far] = frame_bgr[far]
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, color_layer, alpha, 0)


def draw_detections_with_depth(frame, detections, depth_m, danger_dist, warning_dist):
    """Disegna mask + bbox + label colorate per livello pericolo."""
    has_danger, has_warning = False, False

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        mask = det["mask"]

        if mask is None:
            continue

        level, info = classify_person_by_feet(
            mask, depth_m, danger_dist=danger_dist, warning_dist=warning_dist
        )
        det["level"] = level
        det["info"] = info

        if level == "danger":
            color = (0, 0, 255)        # rosso
            fill_alpha = 0.45
            line_thickness = 3
            has_danger = True
        elif level == "warning":
            color = (0, 255, 255)      # giallo
            fill_alpha = 0.30
            line_thickness = 2
            has_warning = True
        else:
            color = (0, 255, 0)        # verde
            fill_alpha = 0.0
            line_thickness = 2

        # fill
        if fill_alpha > 0:
            overlay = frame.copy()
            overlay[mask] = color
            cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, dst=frame)

        # contorni
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if level == "danger":
            cv2.drawContours(frame, contours, -1, (255, 255, 255), line_thickness + 2)
        cv2.drawContours(frame, contours, -1, color, line_thickness)

        # label
        depth_str = ""
        if info and info.get("median_depth") is not None:
            depth_str = f" {info['median_depth']:.1f}m"
        tag = f" [{level.upper()}]" if level != "safe" else ""
        label = f"person {score:.2f}{depth_str}{tag}"
        cv2.putText(
            frame, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.56, color,
            2 if level == "danger" else 1, cv2.LINE_AA,
        )

    return has_danger, has_warning


def draw_alert_banner(frame, text):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, dst=frame)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    cv2.putText(frame, text, ((w - tw) // 2, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)


def draw_warning_banner(frame, text):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (0, 200, 255), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, dst=frame)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    cv2.putText(frame, text, ((w - tw) // 2, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2, cv2.LINE_AA)


def draw_info(frame, lines):
    if not lines:
        return
    pad = 6
    line_h = 22
    box_h = pad * 2 + line_h * len(lines)
    box_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        box_w = max(box_w, tw + 2 * pad)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    for i, line in enumerate(lines):
        y = pad + line_h * (i + 1) - 6
        cv2.putText(frame, line, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()

    # --- carica calibrazione
    if not os.path.exists(args.calib):
        sys.exit(f"[!] Calibrazione non trovata: {args.calib}")
    print(f"[i] Caricamento calibrazione da {args.calib}")
    calibrator = DepthCalibrator(args.calib)

    # --- carica HEFs
    if not os.path.exists(args.hef_yolo):
        sys.exit(f"[!] HEF YOLO non trovato: {args.hef_yolo}")
    if not os.path.exists(args.hef_depth):
        sys.exit(f"[!] HEF SCDepth non trovato: {args.hef_depth}")

    print(f"[i] Caricamento HEF YOLO:  {args.hef_yolo}")
    hef_yolo = HEF(args.hef_yolo)
    print(f"[i] Caricamento HEF Depth: {args.hef_depth}")
    hef_depth = HEF(args.hef_depth)

    with VDevice() as target:
        # configure entrambi i network groups
        ng_yolo = target.configure(hef_yolo)[0]
        ng_depth = target.configure(hef_depth)[0]

        yolo_in_p, yolo_out_p = make_vstream_params(ng_yolo)
        depth_in_p, depth_out_p = make_vstream_params(ng_depth)

        yolo_in_info = ng_yolo.get_input_vstream_infos()[0]
        depth_in_info = ng_depth.get_input_vstream_infos()[0]

        print(f"[i] YOLO  input:  {yolo_in_info.name} {yolo_in_info.shape}")
        for o in ng_yolo.get_output_vstream_infos():
            print(f"        output: {o.name} {o.shape}")
        print(f"[i] Depth input:  {depth_in_info.name} {depth_in_info.shape}")
        for o in ng_depth.get_output_vstream_infos():
            print(f"        output: {o.name} {o.shape}")

        # --- video I/O
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"[!] Impossibile aprire video: {args.video}")
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[i] Video: {W}x{H} @ {fps:.1f} fps, {n_frames} frame")

        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
        )
        if not writer.isOpened():
            sys.exit(f"[!] Impossibile aprire writer: {args.output}")

        # --- main loop
        frame_idx = 0
        printed_debug = False
        t_start = time.time()
        times_yolo = []
        times_depth = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()

            # ----- YOLO inference -----
            yolo_input, yolo_meta = preprocess_yolo(frame)
            t0 = time.time()
            with ng_yolo.activate(), \
                 InferVStreams(ng_yolo, yolo_in_p, yolo_out_p) as ys:
                yolo_raw = ys.infer({yolo_in_info.name: yolo_input})
            times_yolo.append(time.time() - t0)

            # ----- SCDepth inference -----
            depth_input = preprocess_scdepth(frame)
            t0 = time.time()
            with ng_depth.activate(), \
                 InferVStreams(ng_depth, depth_in_p, depth_out_p) as ds:
                depth_raw = ds.infer({depth_in_info.name: depth_input})
            times_depth.append(time.time() - t0)

            if args.debug_outputs and not printed_debug:
                print("\n=== YOLO outputs ===")
                for n, a in yolo_raw.items():
                    print(f"  {n}: {np.asarray(a).shape}  dtype={np.asarray(a).dtype}")
                print("=== Depth outputs ===")
                for n, a in depth_raw.items():
                    print(f"  {n}: {np.asarray(a).shape}  dtype={np.asarray(a).dtype}")
                printed_debug = True

            # ----- decode YOLO -----
            try:
                detections = decode_yolov8_seg(
                    {n: np.asarray(a) for n, a in yolo_raw.items()},
                    yolo_meta,
                    person_score_thr=args.person_score_thr,
                    nms_thr=args.nms_thr,
                    mask_thr=args.mask_thr,
                    topk=args.topk_per_scale,
                )
            except Exception as e:
                detections = []
                print(f"[!] decoder YOLO fallito frame {frame_idx}: {e}")

            # ----- decode depth → metri -----
            try:
                depth_m = postprocess_scdepth(depth_raw, calibrator, (H, W))
            except Exception as e:
                depth_m = np.full((H, W), 100.0, dtype=np.float32)
                print(f"[!] decoder Depth fallito frame {frame_idx}: {e}")

            # ----- overlay prossimità -----
            display = proximity_overlay(
                display, depth_m, max_distance=args.warning_dist, alpha=PROX_OVERLAY_ALPHA
            )

            # ----- disegna persone con classificazione depth -----
            has_danger, has_warning = False, False
            if detections:
                has_danger, has_warning = draw_detections_with_depth(
                    display, detections, depth_m,
                    danger_dist=args.danger_dist,
                    warning_dist=args.warning_dist,
                )

            # ----- banner -----
            if has_danger:
                draw_alert_banner(
                    display,
                    f"!! ALERT: PERSONA < {args.danger_dist:.1f}m !!"
                )
            elif has_warning:
                draw_warning_banner(
                    display,
                    f"ATTENZIONE: PERSONA < {args.warning_dist:.1f}m"
                )

            # ----- info -----
            avg_yolo_ms = 1000 * np.mean(times_yolo[-30:])
            avg_depth_ms = 1000 * np.mean(times_depth[-30:])
            d_min = float(np.min(depth_m[depth_m > 0])) if (depth_m > 0).any() else 0.0
            d_med = float(np.median(depth_m[depth_m > 0])) if (depth_m > 0).any() else 0.0
            draw_info(display, [
                f"frame {frame_idx}/{n_frames}  persone: {len(detections)}",
                f"yolo: {avg_yolo_ms:.0f} ms  |  depth: {avg_depth_ms:.0f} ms",
                f"depth: min {d_min:.1f}m  med {d_med:.1f}m  "
                f"|  D<{args.danger_dist:.1f}  W<{args.warning_dist:.1f}",
            ])

            writer.write(display)
            if args.display:
                cv2.imshow("Hailo dual: YOLO-seg + SCDepthV3", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    fps_e2e = frame_idx / elapsed if elapsed > 0 else 0.0
    print(f"\n[i] Done. {frame_idx} frame in {elapsed:.1f}s ({fps_e2e:.2f} fps end-to-end)")
    print(f"[i] YOLO  medio: {1000*np.mean(times_yolo):.1f} ms/frame")
    print(f"[i] Depth medio: {1000*np.mean(times_depth):.1f} ms/frame")
    print(f"[i] Output salvato in: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
