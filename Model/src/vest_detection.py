# model/src/vest_detection.py
from typing import List, Tuple, Optional, Dict, Any, Union
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# ---------- 내부 유틸 ----------
def _ensure_min_short_side(bgr, min_ss: int = 640):
    h, w = bgr.shape[:2]
    ss = min(h, w)
    if ss >= min_ss:
        return bgr, 1.0
    s = float(min_ss) / float(ss)
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC), s

def _cxcywh_to_xyxy(x, y, w, h):
    return [x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0]

def _clip_xyxy(xyxy, W, H):
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(W - 1), x1))
    y1 = max(0.0, min(float(H - 1), y1))
    x2 = max(0.0, min(float(W - 1), x2))
    y2 = max(0.0, min(float(H - 1), y2))
    return [x1, y1, x2, y2]

def _iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

def _rgb_heuristic_mask(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    cond1  = (g > 1.05 * b) & (r > 0.75 * g)   # 형광조끼 계열
    bright = (0.2126 * r + 0.7152 * g + 0.0722 * b) > 80.0
    return ((cond1 & bright).astype(np.uint8) * 255)

# ---------- 공개 함수 ----------
def make_vest_client(api_url: str, api_key: str, conf_thresh: float, iou_thresh: float):
    cli = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    cli.configure(InferenceConfiguration(confidence_threshold=float(conf_thresh),
                                         iou_threshold=float(iou_thresh)))
    return cli

def infer_vest_boxes(
    client: InferenceHTTPClient,
    frame_bgr,
    model_id: str,
    label_name: Optional[Union[str, List[str]]] = "Vest",
    post_conf_thr: float = 0.60,
    min_short_side: int = 640
) -> List[Tuple[list, float, str]]:
    """
    Roboflow API 호출 → [(xyxy, conf, class), ...] 반환
    - label_name: str 또는 list. 빈 문자열/None이면 라벨 필터 해제.
    - 유사어(safety-vest, hi-vis 등) 자동 허용.
    """
    H, W = frame_bgr.shape[:2]
    send, s = _ensure_min_short_side(frame_bgr, min_short_side)
    resp = client.infer(send, model_id=model_id)
    preds = resp.get("predictions", []) or []
    out: List[Tuple[list, float, str]] = []

    # 라벨 정규화 및 유사어 처리
    def _norm(x: str) -> str:
        return x.strip().lower().replace("_", "-").replace(" ", "")

    if isinstance(label_name, str) or label_name is None:
        wants = [] if (label_name is None or _norm(label_name) in ("", "none")) else [_norm(label_name)]
    else:
        wants = [_norm(x) for x in label_name]

    VEST_ALIASES = {
        "vest", "safety-vest", "safetyvest",
        "hi-vis", "hivis", "high-visibility-vest",
        "reflective-vest"
    }

    def _matches(cname_norm: str) -> bool:
        if not wants:  # 필터 해제
            return True
        if any(w in ("vest", "safety-vest", "safetyvest") for w in wants):
            return (cname_norm in VEST_ALIASES) or ("vest" in cname_norm)
        return cname_norm in set(wants)

    for p in preds:
        cname_raw = (p.get("class") or p.get("class_name") or "")
        cname = _norm(cname_raw)
        conf  = float(p.get("confidence", 0.0))
        if not _matches(cname):
            continue
        if conf < float(post_conf_thr):
            continue

        cx, cy, pw, ph = map(float, (p["x"], p["y"], p["width"], p["height"]))
        xyxy = _cxcywh_to_xyxy(cx, cy, pw, ph)
        if s != 1.0:
            xyxy = [v / s for v in xyxy]
        xyxy = _clip_xyxy(xyxy, W, H)
        out.append((xyxy, conf, cname_raw))
    return out

def vest_color_ratio(bgr_roi):
    """형광 조끼(주황/노랑/라임) 영역 비율 추정."""
    if bgr_roi is None or bgr_roi.size == 0:
        return 0.0, None

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.createCLAHE(2.0, (8, 8)).apply(v)
    hsv = cv2.merge([h, s, v])

    # 주황(5~25), 노랑(25~45), 라임/연두(45~85)
    m_orange = cv2.inRange(hsv, (5,  60,  80), (25, 255, 255))
    m_yellow = cv2.inRange(hsv, (25, 60,  80), (45, 255, 255))
    m_lime   = cv2.inRange(hsv, (45, 50,  70), (85, 255, 255))
    mask = cv2.bitwise_or(m_orange, cv2.bitwise_or(m_yellow, m_lime))

    # RGB 밝기/채도 휴리스틱 추가
    mask_rgb = _rgb_heuristic_mask(bgr_roi)
    mask = cv2.bitwise_or(mask, mask_rgb)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, 1)

    H, W = mask.shape[:2]
    ratio = float((mask > 0).sum()) / float(max(1, H * W))
    return ratio, mask

def geom_ok(x1, y1, x2, y2, W, H,
            area_min_frac=0.002, area_max_frac=0.25,
            aspect_min=0.35, aspect_max=2.20):
    w = max(1, int(x2 - x1)); h = max(1, int(y2 - y1))
    area_frac = (w * h) / float(W * H)
    ar = w / float(h)
    if area_frac < area_min_frac:
        return False, "too_small"
    if area_frac > area_max_frac:
        return False, "too_big"
    if not (aspect_min <= ar <= aspect_max):
        return False, f"bad_aspect({ar:.2f})"
    return True, ""

def center_in_person_torso(x1, y1, x2, y2, person_boxes,
                           torso_low=0.20, torso_high=0.80):
    if not person_boxes:
        return True
    cx = 0.5 * (x1 + x2); cy = 0.5 * (y1 + y2)
    for (px1, py1, px2, py2), _ in person_boxes:
        ph = max(1.0, py2 - py1)
        ty1 = py1 + torso_low * ph; ty2 = py1 + torso_high * ph
        if (px1 <= cx <= px2) and (ty1 <= cy <= ty2):
            return True
    return False

def match_vest_to_person(person_xyxy, kept_vests,
                         torso_low=0.20, torso_high=0.80, iou_thr=0.05):
    x1, y1, x2, y2 = person_xyxy
    best_c = 0.0; best_r = 0.0; found = False
    ph = max(1.0, y2 - y1)
    ty1 = y1 + torso_low * ph; ty2 = y1 + torso_high * ph
    for vxyxy, c, ratio in kept_vests:
        if _iou_xyxy(person_xyxy, vxyxy) < iou_thr:
            continue
        vcx = 0.5 * (vxyxy[0] + vxyxy[2])
        vcy = 0.5 * (vxyxy[1] + vxyxy[3])
        if (x1 <= vcx <= x2) and (ty1 <= vcy <= ty2):
            if c > best_c:
                best_c = c; best_r = ratio; found = True
    return (found, (best_c if found else None), (best_r if found else None))
