"""
API-only 조끼(jacket) 감지 파이프라인 (VIDEO VERSION)
- Roboflow Serverless API만 사용 (로컬 pose 불필요)
- 색상/기하학 필터 + (선택) person 게이팅
- 폴더 내 모든 '동영상'을 재귀 처리하여 주석 비디오(mp4) 출력 + JSONL 로깅
필요:
    pip install inference-sdk opencv-python
"""

from typing import List, Tuple, Optional
import os, time, json, glob, math
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from config import *

# ===================== 설정 =====================

VEST_API = API_KEY
VEST_MODEL_ID    = "construction-ppe-rdhzo/3" # 조끼 모델 (RF Universe 프로젝트/버전)
PERSON_MODEL_ID  = ""                                  # 예: "coco-person-yolov8/1" (모르면 빈 문자열 -> 비활성)
LABEL            = "Vest"                              # "None" 혹은 "" 이면 라벨 제한 해제, 특정 라벨만 원하면 "jacket" 등

VIDEO_DIR        = "Video"
OUT_DIR          = "vest__detection"

# API/후처리 하이퍼파라미터
MIN_SHORT_SIDE   = 640
API_MIN_CONF     = 0.25   # 서버 측
API_IOU_THR      = 0.50
POST_CONF_THR    = 0.6   # 후단 필터(더 타이트하게 하려면 0.6 ~ 0.7)
AREA_MIN_FRAC    = 0.002  # 박스 면적이 이미지 대비 너무 작으면 drop
AREA_MAX_FRAC    = 0.25   # 너무 큰 박스(drop: 천장/배경 큰 오탐 방지)
ASPECT_MIN       = 0.35   # w/h 하한
ASPECT_MAX       = 2.2    # w/h 상한

# 색 필터(형광 조끼) - ratio 임계 완화/강화 조절
VEST_PIXEL_RATIO_THR = 0.02   # 박스 내부에서 형광색 픽셀 비율(0.01~0.04 사이로 튜닝)

# 사람 게이트(선택) - 사람 bbox의 세로 20~80% 밴드를 상체로 간주
TORSO_RATIO_LOW  = 0.20
TORSO_RATIO_HIGH = 0.80

# 동영상 처리 옵션
FRAME_STRIDE     = 20        # n프레임마다 처리 (1이면 모든 프레임 처리)
SEC_STRIDE       = None     # 초 단위 샘플링 (예: 0.5 -> 0.5초마다 한 번). 지정 시 FRAME_STRIDE 대신 사용
MAX_VIDEOS       = None     # 처리할 최대 동영상 수 (디버그용), None이면 전체
SLEEP_BETWEEN    = 0.0      # 프레임마다 sleep (API rate 조절용)

# 디버그/표시
DRAW_ALL_IF_EMPTY = True
LOWER_DEBUG_THR   = 0.10
BOX_THICK         = 2
FONT_SCALE        = 0.6
PRINT_EVERY_N_FR  = 30

# ===================== 안전 IO (한글/긴 경로) =====================
def _ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _rel(path: str, base: str):
    try: return os.path.relpath(path, base)
    except Exception: return os.path.basename(path)

# ===================== 유틸 =====================
def _ensure_min_short_side(bgr, min_ss=640):
    h, w = bgr.shape[:2]
    ss = min(h, w)
    if ss >= min_ss: return bgr, 1.0
    s = float(min_ss) / float(ss)
    return cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC), s

def _cxcywh_to_xyxy(x, y, w, h):
    return [x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0]

def _clip_xyxy(xyxy, W, H):
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(W-1), x1))
    y1 = max(0.0, min(float(H-1), y1))
    x2 = max(0.0, min(float(W-1), x2))
    y2 = max(0.0, min(float(H-1), y2))
    return [x1, y1, x2, y2]

def _list_videos(root: str):
    exts = ("*.mp4","*.avi","*.mov","*.mkv","*.wmv","*.m4v")
    files=[]
    for e in exts: files += glob.glob(os.path.join(root, "**", e), recursive=True)
    files.sort()
    return files

# ===================== 색상 마스크(형광 조끼) =====================
def _rgb_heuristic_mask(bgr):
    b,g,r = cv2.split(bgr.astype(np.float32))
    cond1  = (g > 1.05*b) & (r > 0.75*g)   # 초록이 파랑보다 크고, 빨강도 어느정도
    bright = (0.2126*r + 0.7152*g + 0.0722*b) > 80.0
    return ((cond1 & bright).astype(np.uint8) * 255)

def vest_color_ratio(bgr_roi):
    if bgr_roi is None or bgr_roi.size == 0: return 0.0, None
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v); hsv = cv2.merge([h,s,v])

    ranges = [
        (np.array([15, 40, 50]), np.array([45,255,255])),  # yellow
        (np.array([30, 30, 50]), np.array([90,255,255])),  # lime/greenish
        (np.array([ 0, 60, 50]), np.array([20,255,255])),  # orange-ish
    ]
    mask_total = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask_total = cv2.bitwise_or(mask_total, cv2.inRange(hsv, lo, hi))
    mask_total = cv2.bitwise_or(mask_total, _rgb_heuristic_mask(bgr_roi))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, k, 1)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN,  k, 1)

    H,W = mask_total.shape[:2]
    ratio = float((mask_total > 0).sum()) / float(max(1, H*W))
    return ratio, mask_total

# ===================== API 클라이언트 / 추론 =====================
def make_client(api_url: str, api_key: str) -> InferenceHTTPClient:
    cli = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    cli.configure(InferenceConfiguration(confidence_threshold=float(API_MIN_CONF),
                                         iou_threshold=float(API_IOU_THR)))
    return cli

def infer_boxes(client: InferenceHTTPClient, frame_bgr, model_id: str,
                label_name: Optional[str] = None,
                min_conf: float = POST_CONF_THR,
                min_short_side: int = MIN_SHORT_SIDE) -> List[Tuple[list, float, str]]:
    """
    반환: [(xyxy, conf, cname), ...]
    - label_name in {"", None, "none", "None"}이면 라벨 제한 없음
    """
    H,W = frame_bgr.shape[:2]
    send, s = _ensure_min_short_side(frame_bgr, min_short_side)
    resp = client.infer(send, model_id=model_id)
    preds = resp.get("predictions", []) or []
    out=[]
    want_label = None
    if label_name is not None and str(label_name).strip().lower() not in ("", "none"):
        want_label = str(label_name).strip().lower()

    for p in preds:
        cname = (p.get("class") or p.get("class_name") or "").strip().lower()
        conf  = float(p.get("confidence", 0.0))
        if want_label and cname != want_label:
            continue
        if conf < float(min_conf):
            continue
        cx,cy,pw,ph = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
        xyxy = _cxcywh_to_xyxy(cx,cy,pw,ph)
        if s != 1.0: xyxy = [v/s for v in xyxy]
        xyxy = _clip_xyxy(xyxy, W, H)
        out.append((xyxy, conf, cname))
    return out

# ===================== 후처리(기하학/색/사람 게이트) =====================
def geom_ok(x1,y1,x2,y2, W,H):
    w = max(1, int(x2-x1)); h = max(1, int(y2-y1))
    area_frac = (w*h) / float(W*H)
    ar = w / float(h)
    if area_frac < AREA_MIN_FRAC: return False, "too_small"
    if area_frac > AREA_MAX_FRAC: return False, "too_big"
    if not (ASPECT_MIN <= ar <= ASPECT_MAX): return False, f"bad_aspect({ar:.2f})"
    return True, ""

def gate_by_person_center(x1,y1,x2,y2, person_boxes):
    if not person_boxes: return True  # 게이트 비활성화 시 통과
    cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)
    for (px1,py1,px2,py2), _ in person_boxes:
        ph = max(1.0, py2-py1)
        ty1 = py1 + TORSO_RATIO_LOW  * ph
        ty2 = py1 + TORSO_RATIO_HIGH * ph
        if (px1 <= cx <= px2) and (ty1 <= cy <= ty2):
            return True
    return False

# ===================== 비디오 처리 =====================
def _video_out_path(in_path: str, base_dir: str, out_root: str):
    rel = _rel(in_path, base_dir)
    rel_noext = os.path.splitext(rel)[0]
    out_path = os.path.join(out_root, "videos", rel_noext + ".mp4")
    _ensure_parent_dir(out_path)
    return out_path, rel

def _should_process_frame(frame_idx: int, fps: float, sec_stride: Optional[float], frame_stride: int):
    if sec_stride is not None and fps > 0:
        # fps * sec_stride 마다 한 프레임 처리 (가장 가까운 정수 프레임에 스냅)
        step = max(1, int(round(fps * float(sec_stride))))
        return (frame_idx % step) == 0
    else:
        return (frame_idx % max(1, int(frame_stride))) == 0

def run_videos_in_folder():
    client = make_client("https://serverless.roboflow.com", API_KEY)

    os.makedirs(OUT_DIR, exist_ok=True)
    vis_root = os.path.join(OUT_DIR, "videos"); os.makedirs(vis_root, exist_ok=True)
    jsonl_path = os.path.join(OUT_DIR, "detections.jsonl")

    vids = _list_videos(VIDEO_DIR)
    if not vids:
        print(f"[WARN] No videos in: {VIDEO_DIR}"); return
    if isinstance(MAX_VIDEOS, int) and MAX_VIDEOS > 0:
        vids = vids[:MAX_VIDEOS]

    print(f"[INFO] Found {len(vids)} videos under: {VIDEO_DIR}")
    kept_total = 0

    with open(jsonl_path, "a", encoding="utf-8") as jf:
        for vi, vp in enumerate(vids, 1):
            out_path, rel = _video_out_path(vp, VIDEO_DIR, OUT_DIR)
            print(f"\n[{vi}/{len(vids)}] {rel}")

            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                print(f"   [WARN] cannot open video: {vp}")
                continue

            fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                print(f"   [ERROR] cannot open writer: {out_path}")
                cap.release()
                continue

            per_video_kept = 0
            frame_idx = 0
            last_draw = None

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                do_process = _should_process_frame(frame_idx, fps, SEC_STRIDE, FRAME_STRIDE)

                draw = frame.copy()
                kept = []

                if do_process:
                    H,W = frame.shape[:2]
                    # 1) jacket 후보
                    try:
                        jacket_boxes = infer_boxes(client, frame, VEST_MODEL_ID, label_name=LABEL,
                                                   min_conf=POST_CONF_THR, min_short_side=MIN_SHORT_SIDE)
                    except Exception as e:
                        print(f"   [ERROR] jacket API on frame {frame_idx}: {e}")
                        jacket_boxes = []

                    # (선택) 2) person 박스
                    person_boxes=[]
                    if PERSON_MODEL_ID:
                        try:
                            raw_person = infer_boxes(client, frame, PERSON_MODEL_ID, label_name=None,
                                                     min_conf=0.40, min_short_side=MIN_SHORT_SIDE)
                            for (xyxy, conf, cname) in raw_person:
                                if cname == "person":
                                    person_boxes.append((xyxy, conf))
                        except Exception as e:
                            print(f"   [WARN] person API on frame {frame_idx}: {e}")

                    # 3) 필터링 + 그리기
                    for (xyxy, conf, cname) in jacket_boxes:
                        x1,y1,x2,y2 = map(int, _clip_xyxy(xyxy, W, H))

                        ok_g, why = geom_ok(x1,y1,x2,y2, W,H)
                        if not ok_g:
                            if DRAW_ALL_IF_EMPTY:
                                cv2.rectangle(draw,(x1,y1),(x2,y2),(0,255,255),BOX_THICK)
                                cv2.putText(draw, f"drop:{why}", (x1, max(0,y1-6)),
                                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,255,255),2,cv2.LINE_AA)
                            continue

                        roi = frame[y1:y2, x1:x2]
                        ratio, _ = vest_color_ratio(roi)
                        if ratio < VEST_PIXEL_RATIO_THR:
                            if DRAW_ALL_IF_EMPTY:
                                cv2.rectangle(draw,(x1,y1),(x2,y2),(0,255,255),BOX_THICK)
                                cv2.putText(draw, f"drop:color {ratio:.3f}", (x1, max(0,y1-6)),
                                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,255,255),2,cv2.LINE_AA)
                            continue

                        if not gate_by_person_center(x1,y1,x2,y2, person_boxes):
                            if DRAW_ALL_IF_EMPTY:
                                cv2.rectangle(draw,(x1,y1),(x2,y2),(0,255,255),BOX_THICK)
                                cv2.putText(draw, "drop:no_person_gate", (x1, max(0,y1-6)),
                                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,255,255),2,cv2.LINE_AA)
                            continue

                        # keep
                        kept.append(((x1,y1,x2,y2), conf, cname, ratio))
                        cv2.rectangle(draw,(x1,y1),(x2,y2),(0,255,0),BOX_THICK)
                        label_txt = f"vest {conf:.2f} r={ratio:.3f}"
                        (tw,th),_ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2)
                        cv2.rectangle(draw,(x1,max(0,y1-th-6)),(x1+tw+6,y1),(0,255,0),-1)
                        cv2.putText(draw,label_txt,(x1+3,max(0,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,FONT_SCALE,(0,0,0),2,cv2.LINE_AA)

                    # 디버그: 남은게 0이면 낮은 임계로 전 예측 그려보기
                    if not kept and DRAW_ALL_IF_EMPTY:
                        try:
                            resp = client.infer(frame, model_id=VEST_MODEL_ID)
                            preds = resp.get("predictions", []) or []
                            for p in preds:
                                conf = float(p.get("confidence",0.0))
                                if conf < LOWER_DEBUG_THR: continue
                                cx,cy,pw,ph = float(p["x"]),float(p["y"]),float(p["width"]),float(p["height"])
                                xyxy = _cxcywh_to_xyxy(cx,cy,pw,ph)
                                x1,y1,x2,y2 = map(int, _clip_xyxy(xyxy, W, H))
                                cv2.rectangle(draw,(x1,y1),(x2,y2),(128,128,128),BOX_THICK)
                        except Exception:
                            pass

                    # JSONL 로깅
                    t_sec = frame_idx / float(fps if fps > 0 else 30.0)
                    for (x1,y1,x2,y2), conf, cname, ratio in kept:
                        row = {
                            "source": "video",
                            "video_rel": rel,
                            "video_abs": os.path.abspath(vp),
                            "frame_idx": int(frame_idx),
                            "time_sec": round(float(t_sec), 3),
                            "bbox_xyxy": [int(x1),int(y1),int(x2),int(y2)],
                            "score": round(float(conf), 6),
                            "class": cname,
                            "color_ratio": round(float(ratio), 6)
                        }
                        jf.write(json.dumps(row, ensure_ascii=False) + "\n")

                    kept_total += len(kept)
                    per_video_kept += len(kept)
                    last_draw = draw

                    if SLEEP_BETWEEN > 0:
                        time.sleep(SLEEP_BETWEEN)
                else:
                    # 처리하지 않는 프레임은 마지막 주석 결과가 있으면 그걸 재사용, 없으면 원본
                    if last_draw is not None:
                        draw = last_draw
                    else:
                        draw = frame

                writer.write(draw)

                frame_idx += 1
                if frame_idx % max(1, PRINT_EVERY_N_FR) == 0:
                    if total > 0:
                        print(f"   [FRAMES] {frame_idx}/{total} (kept so far: {per_video_kept})")
                    else:
                        print(f"   [FRAMES] {frame_idx} (kept so far: {per_video_kept})")

            cap.release()
            writer.release()
            print(f"   [SAVE] {out_path} (video_kept={per_video_kept})")

    print("[DONE] kept_total:", kept_total)

# ===================== 엔트리 포인트 =====================
if __name__ == "__main__":
    run_videos_in_folder()
