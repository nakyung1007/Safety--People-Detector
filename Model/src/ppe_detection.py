from typing import List, Tuple, Optional, Dict, Union
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# ----------------- 내부 유틸 -----------------
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

def _area(xyxy):
    return max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])

def _inter(xy1, xy2):
    x1 = max(xy1[0], xy2[0]); y1 = max(xy1[1], xy2[1])
    x2 = min(xy1[2], xy2[2]); y2 = min(xy1[3], xy2[3])
    if x2 <= x1 or y2 <= y1:
        return [0, 0, 0, 0], 0.0
    return [x1, y1, x2, y2], (x2 - x1) * (y2 - y1)

def _norm_label(s: str) -> str:
    return s.strip().lower().replace("_", "-").replace(" ", "")

def _rgb_heuristic_mask(bgr):
    # 밝은 노/연녹/라임 계열 강조 + 충분한 밝기
    b, g, r = cv2.split(bgr.astype(np.float32))
    cond1  = (g > 1.05 * b) & (r > 0.75 * g)
    bright = (0.2126 * r + 0.7152 * g + 0.0722 * b) > 80.0
    return ((cond1 & bright).astype(np.uint8) * 255)

# ----------------- 클라이언트 -----------------
def make_ppe_client(api_url: str, api_key: str,
                    conf_thresh: float, iou_thresh: float,
                    model_id: Optional[str] = None) -> InferenceHTTPClient:
    cli = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    cli.configure(InferenceConfiguration(
        confidence_threshold=float(conf_thresh),
        iou_threshold=float(iou_thresh)
    ))
    if model_id:
        cli._model_id = model_id  # 편의상 저장
    return cli

# ----------------- 색/형상/매칭 유틸 (Vest) -----------------
def vest_color_ratio(bgr_roi, mode: str = "hivis"):
    """
    mode:
      - "hivis": 주황/노랑/라임 위주(기본, FP 적음)
      - "broad": 빨강/파랑까지 허용(색 다양한 조끼를 쓸 때)
    """
    if bgr_roi is None or bgr_roi.size == 0:
        return 0.0, None

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

    hivis_ranges = [
        (np.array([  5,  80,  80]), np.array([ 20, 255, 255])),  # orange
        (np.array([ 20,  80,  80]), np.array([ 45, 255, 255])),  # yellow
        (np.array([ 45,  60,  70]), np.array([ 90, 255, 255])),  # lime/green
    ]
    broad_extra = [
        (np.array([  0,  80,  80]), np.array([ 10, 255, 255])),  # red low
        (np.array([170,  80,  80]), np.array([180, 255, 255])),  # red high
        (np.array([ 95,  80,  80]), np.array([130, 255, 255])),  # blue
    ]

    ranges = list(hivis_ranges)
    if str(mode).lower() == "broad":
        ranges += broad_extra

    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    # RGB 휴리스틱(밝은 노/녹 계열) 추가
    mask = cv2.bitwise_or(mask, _rgb_heuristic_mask(bgr_roi))

    # 노이즈 제거
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
    if area_frac < area_min_frac: return False, "too_small"
    if area_frac > area_max_frac: return False, "too_big"
    if not (aspect_min <= ar <= aspect_max): return False, f"bad_aspect({ar:.2f})"
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
    """
    더 엄격한 매칭:
      - IoU >= iou_thr
      - 중심점이 torso band 안
      - vest 박스의 'torso band 내 포함비율' >= 0.6
      - 사람 대비 폭/높이 비율이 합리적인 범위
    """
    x1, y1, x2, y2 = person_xyxy
    best_c = 0.0; best_r = 0.0; found = False
    pw = max(1.0, x2 - x1); ph = max(1.0, y2 - y1)
    ty1 = y1 + torso_low * ph; ty2 = y1 + torso_high * ph
    torso_band = [x1, ty1, x2, ty2]

    for vxyxy, c, ratio in kept_vests:
        if _iou_xyxy(person_xyxy, vxyxy) < iou_thr:
            continue

        vcx = 0.5 * (vxyxy[0] + vxyxy[2]); vcy = 0.5 * (vxyxy[1] + vxyxy[3])
        if not ((x1 <= vcx <= x2) and (ty1 <= vcy <= ty2)):
            continue

        _, inter_area = _inter(vxyxy, torso_band)
        vest_area = _area(vxyxy)
        if vest_area <= 0:
            continue
        contain_ratio = inter_area / vest_area
        if contain_ratio < 0.60:
            continue

        vw = max(1.0, vxyxy[2] - vxyxy[0]); vh = max(1.0, vxyxy[3] - vxyxy[1])
        w_frac = vw / pw
        h_frac = vh / ph
        if not (0.25 <= w_frac <= 0.95 and 0.20 <= h_frac <= 0.75):
            continue

        if c > best_c:
            best_c = c; best_r = ratio; found = True

    return (found, (best_c if found else None), (best_r if found else None))

# ----------------- 단일 API 추론 -----------------
_HELMET_ALIASES = {
    "helmet", "hardhat", "hard-hat", "safety-helmet",
    "helmeton", "helmet-on", "hardhaton", "hard-haton",
    "helmets", "hardhats",
}
_VEST_ALIASES = {
    "vest", "safety-vest", "safetyvest", "safety-vests", "safetyvests",
    "hi-vis", "hivis", "high-visibility-vest", "highvisibilityvest",
    "reflective-vest", "reflectivevest", "reflective-jacket", "safety-jacket",
    "veston", "vest-on", "safetyveston", "safety-vest-on", "harness", "safety-harness", "body-harness"
}

def infer_ppe(client: InferenceHTTPClient,
              frame_bgr,
              model_id: str,
              post_conf_thr: float = 0.25,
              min_short_side: int = 640
) -> Dict[str, List[Tuple[list, float, str]]]:
    """
    단일 모델 호출 → {'helmets': [(xyxy, conf, class), ...],
                    'vests'  : [(xyxy, conf, class), ...]}
    """
    H, W = frame_bgr.shape[:2]
    send, s = _ensure_min_short_side(frame_bgr, min_short_side)
    resp = client.infer(send, model_id=model_id)
    preds = resp.get("predictions", []) or []

    helmets: List[Tuple[list, float, str]] = []
    vests:   List[Tuple[list, float, str]] = []

    for p in preds:
        cname_raw = (p.get("class") or p.get("class_name") or "")
        cname = _norm_label(cname_raw)
        conf  = float(p.get("confidence", 0.0))
        if conf < float(post_conf_thr):
            continue

        cx, cy, pw, ph = map(float, (p["x"], p["y"], p["width"], p["height"]))
        xyxy = _cxcywh_to_xyxy(cx, cy, pw, ph)
        if s != 1.0:
            xyxy = [v / s for v in xyxy]
        xyxy = _clip_xyxy(xyxy, W, H)

        if (cname in _HELMET_ALIASES) or ("helmet" in cname) or ("hardhat" in cname):
            helmets.append((xyxy, conf, cname_raw))
        elif (cname in _VEST_ALIASES) or ("vest" in cname) or ("hivis" in cname):
            vests.append((xyxy, conf, cname_raw))

    return {"helmets": helmets, "vests": vests}

# ----------------- 사람별 판정(헬멧) -----------------
def person_has_helmet(
    person_xyxy,
    helmet_boxes: List[Tuple[list, float, str]],
    top_ratio: float = 0.60,
    iou_thr: Optional[float] = 0.05,
    min_cover_frac: float = 0.35,               # 머리 밴드와의 겹침(헬멧 면적 기준)
    eye_margin_frac: float = 0.02,              # 눈/코/귀 라인보다 얼마나 위에 있어야 하는지
    size_h_frac_range: Tuple[float, float] = (0.06, 0.22),  # 헬멧 높이 / 사람 높이
    size_w_frac_range: Tuple[float, float] = (0.08, 0.35),  # 헬멧 폭   / 사람 폭
    ar_range: Tuple[float, float] = (0.6, 1.8),             # 종횡비 범위
    kpt_xy: Optional[np.ndarray] = None,
    kpt_conf: Optional[np.ndarray] = None,
    kpt_thr: float = 0.30
) -> Tuple[bool, Optional[float]]:
    """
    - 사람 박스의 '머리 밴드' 내부(키포인트 있으면 어깨선 위/눈-코-귀 위)에서만 허용
    - 머리 밴드 포함비율(>= min_cover_frac) + 크기/종횡비 제약
    - iou_thr가 주어지면 IoU도 추가로 체크(0.03~0.05 권장)
    """
    x1, y1, x2, y2 = person_xyxy
    pw = max(1.0, x2 - x1); ph = max(1.0, y2 - y1)

    # 1) 머리 밴드 계산 (키포인트 우선)
    head_bottom = y1 + top_ratio * ph
    eyes_line = None
    if kpt_xy is not None and kpt_conf is not None and len(kpt_xy) >= 7:
        # 어깨선 기반 head_bottom 보정
        shoulders = []
        for idx in (5, 6):  # L/R shoulder
            if kpt_conf[idx] is not None and float(kpt_conf[idx]) >= kpt_thr:
                shoulders.append(float(kpt_xy[idx][1]))
        if shoulders:
            head_bottom = min(shoulders) - 0.03 * ph  # 어깨보다 살짝 위까지

        # 눈/코/귀 라인
        head_pts = []
        for idx in (0, 1, 2, 3, 4):  # nose, eyes, ears
            if kpt_conf[idx] is not None and float(kpt_conf[idx]) >= kpt_thr:
                head_pts.append(float(kpt_xy[idx][1]))
        if head_pts:
            eyes_line = min(head_pts)

    head_band = [x1, y1, x2, head_bottom]

    best = 0.0; found = False
    for hxyxy, score, _ in helmet_boxes:
        if iou_thr is not None and iou_thr > 0.0:
            if _iou_xyxy(person_xyxy, hxyxy) < iou_thr:
                continue

        hx1, hy1, hx2, hy2 = hxyxy
        hx = 0.5 * (hx1 + hx2); hy = 0.5 * (hy1 + hy2)

        # 사람 박스 내부 & 머리 밴드 내부
        if not (x1 <= hx <= x2 and y1 <= hy <= head_bottom):
            continue

        # 눈/코/귀 라인 기준(존재 시)
        if eyes_line is not None and not (hy <= eyes_line - eye_margin_frac * ph):
            continue

        # 머리 밴드 포함비율
        _, inter_area = _inter(hxyxy, head_band)
        ha = _area(hxyxy)
        if ha <= 0 or (inter_area / ha) < float(min_cover_frac):
            continue

        # 크기/종횡비
        hw = max(1.0, hx2 - hx1); hh = max(1.0, hy2 - hy1)
        h_frac = hh / ph
        w_frac = hw / pw
        ar = hw / hh
        if not (size_h_frac_range[0] <= h_frac <= size_h_frac_range[1]):
            continue
        if not (size_w_frac_range[0] <= w_frac <= size_w_frac_range[1]):
            continue
        if not (ar_range[0] <= ar <= ar_range[1]):
            continue

        if score > best:
            best = score; found = True

    return (found, (best if found else None))

# =========================
# [ADD-ON] wide color support for vests
# 붙여넣기 위치: ppe_detection.py 맨 아래
# =========================

def _retroreflective_mask(hsv, sat_max=60, val_min=200):
    """
    저채도·고밝기(흰색/은색 반사띠) 마스크.
    sat_max를 낮추면 더 ‘진짜 흰색/은색’만, val_min을 올리면 더 밝은 것만 잡음.
    """
    H, S, V = cv2.split(hsv)
    m = ((S <= sat_max) & (V >= val_min)).astype(np.uint8) * 255
    return m

def vest_color_ratio(
    bgr_roi,
    mode: str = "hivis",
    extra_hsv_ranges: Optional[List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]] = None
):
    """
    mode:
      - "hivis" : 주황/노랑/라임 (기본, FP 적음)
      - "broad" : hivis + 빨강/파랑
      - "all"   : broad + 청록/보라/분홍 + 반사띠(흰/은색)
    extra_hsv_ranges: [(loHSV, hiHSV), ...] 형태로 원하는 색대를 추가로 넘겨 커스터마이즈 가능
                      예) [((100,80,80),(110,255,255))]  # 진한 파랑만 더 잡기
    반환: (ratio, mask)
    """
    if bgr_roi is None or bgr_roi.size == 0:
        return 0.0, None

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

    # --- 기본 하이비즈 대역(주황/노랑/라임) ---
    ranges = [
        (np.array([  5,  80,  80]), np.array([ 20, 255, 255])),  # orange
        (np.array([ 20,  80,  80]), np.array([ 45, 255, 255])),  # yellow
        (np.array([ 45,  60,  70]), np.array([ 90, 255, 255])),  # lime/green
    ]

    mode_lc = str(mode).lower().strip()
    if mode_lc in ("broad", "all"):
        # 빨강(양끝) + 파랑
        ranges += [
            (np.array([  0,  80,  80]), np.array([ 10, 255, 255])),  # red low
            (np.array([170,  80,  80]), np.array([180, 255, 255])),  # red high
            (np.array([ 95,  70,  70]), np.array([130, 255, 255])),  # blue
        ]

    if mode_lc == "all":
        # 청록/하늘/보라/분홍(마젠타) 추가
        ranges += [
            (np.array([ 85,  60,  70]), np.array([100, 255, 255])),  # cyan/teal
            (np.array([130,  60,  70]), np.array([150, 255, 255])),  # purple/violet
            (np.array([140,  60,  80]), np.array([169, 255, 255])),  # magenta/pink
        ]

    # 사용자 추가 대역
    if extra_hsv_ranges:
        for lo, hi in extra_hsv_ranges:
            ranges.append((np.array(lo), np.array(hi)))

    # 색 마스크 조립
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    # 기존 밝은 노/연녹 휴리스틱 있으면 더함
    try:
        mask = cv2.bitwise_or(mask, _rgb_heuristic_mask(bgr_roi))
    except NameError:
        pass

    # 반사띠(흰/은색) — ALL 모드에서만 추가
    if mode_lc == "all":
        mask = cv2.bitwise_or(mask, _retroreflective_mask(hsv, sat_max=60, val_min=200))

    # 모폴로지로 노이즈 정리
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, 1)

    H, W = mask.shape[:2]
    ratio = float((mask > 0).sum()) / float(max(1, H * W))
    return ratio, mask
