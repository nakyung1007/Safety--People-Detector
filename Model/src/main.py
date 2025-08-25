# scripts/quick_run.py
import os, sys, glob, time, json
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from ultralytics import YOLO

# --- 경로/설정 로딩 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    VIDEO_DIR, OUT_DIR, MODEL_PATH, TRACKER_YAML, IMG_SIZE, DET_CONF,
    DEVICE, KPT_CONF, HELMET_CONFIG, VEST_CONFIG
)

# --- 외부 모듈 ---
from tracker import Tracker
from People_detection import get_person_part, person_visible, draw_person
from helmet_Detection import make_helmet_client, infer_helmet_boxes, person_has_helmet
from vest_detection import (
    make_vest_client, infer_vest_boxes, vest_color_ratio,
    geom_ok, center_in_person_torso, match_vest_to_person
)

# ------------------ 추가: 라벨/박스 유틸 & 하네스 alias ------------------
HARNESS_ALIASES = {
    "harness", "safety-harness", "body-harness", "safetyharness",
    "fall-arrest-harness", "fullbody-harness", "full-body-harness",
    "belt-harness"
}

def _norm_label(s: str) -> str:
    return str(s).strip().lower().replace("_", "-").replace(" ", "")

def _norm_triplets(boxes, default_cls: str = "") -> List[Tuple[list, float, str]]:
    """
    다양한 응답 포맷을 (xyxy, conf, class)로 표준화.
    - (xyxy, conf) 또는 (xyxy, conf, class) 또는 dict 지원.
    """
    out: List[Tuple[list, float, str]] = []
    if not boxes:
        return out
    for b in boxes:
        if isinstance(b, dict):
            xyxy = b.get("xyxy") or b.get("bbox") or b.get("box")
            conf = b.get("confidence") or b.get("conf") or b.get("score") or 0.0
            cls  = b.get("class") or b.get("class_name") or default_cls
            if xyxy is not None:
                out.append((list(map(float, xyxy)), float(conf), str(cls)))
        elif isinstance(b, (list, tuple)):
            if len(b) == 3:
                xyxy, conf, cls = b
                out.append((list(map(float, xyxy)), float(conf), str(cls)))
            elif len(b) == 2:
                xyxy, conf = b
                out.append((list(map(float, xyxy)), float(conf), str(default_cls)))
    return out

# ------------------ 한 비디오 처리 ------------------
def process_one_video(video_path: str,
                      jf,
                      model: YOLO,
                      tracker: "Tracker",
                      helmet_client,
                      vest_client,
                      max_frames: int,
                      helmet_every: int,
                      vest_every: int):
    print(f"[RUN] {video_path}", flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] 비디오 열기 실패: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(os.path.join(OUT_DIR, "video"), exist_ok=True)
    out_path = os.path.join(OUT_DIR, "video", f"{base}__quick.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print(f"[ERR] writer 열기 실패: {out_path}")
        cap.release()
        return

    # --- 헬멧 설정 ---
    h_min_conf        = max(0.0, float(HELMET_CONFIG.get("conf_thresh", 0.25)))
    helmet_label      = str(HELMET_CONFIG.get("label_name", "")).strip()
    h_min_short_side  = int(HELMET_CONFIG.get("min_short_side", 640))
    h_carry_ttl       = int(HELMET_CONFIG.get("carry_ttl", 60))
    h_top_ratio       = float(HELMET_CONFIG.get("top_ratio", 0.60))

    # --- 조끼 설정 ---
    v_post_conf       = float(VEST_CONFIG.get("post_conf_thr", 0.60))
    v_min_short_side  = int(VEST_CONFIG.get("min_short_side", 640))
    v_ratio_thr       = float(VEST_CONFIG.get("ratio_thr", 0.02))
    v_area_min_frac   = float(VEST_CONFIG.get("area_min_frac", 0.002))
    v_area_max_frac   = float(VEST_CONFIG.get("area_max_frac", 0.25))
    v_aspect_min      = float(VEST_CONFIG.get("aspect_min", 0.35))
    v_aspect_max      = float(VEST_CONFIG.get("aspect_max", 2.20))
    v_torso_low       = float(VEST_CONFIG.get("torso_low", 0.20))
    v_torso_high      = float(VEST_CONFIG.get("torso_high", 0.80))
    v_carry_ttl       = int(VEST_CONFIG.get("carry_ttl", 60))

    # 라벨 필터 전략 (하네스 포함)
    vest_label_cfg = str(VEST_CONFIG.get("label_name", "")).strip().lower()
    if vest_label_cfg in ("", "any", "all", "*"):
        vest_label_for_call = ""  # 필터 해제(서버가 아는 모든 'vest/harness' 반환)
    else:
        vest_label_for_call = [
            "vest", "safety-vest", "safetyvest",
            "hi-vis", "high-visibility-vest", "reflective-vest",
            # harness 계열도 조끼로 취급
            "harness", "safety-harness", "body-harness", "safetyharness",
            "fall-arrest-harness", "full-body-harness"
        ]

    frame_idx = 0
    kept_people_frames = 0
    t0 = time.time()

    # ---- FPS/지연 오버레이용 상태 ----
    ema = None
    t_prev = time.time()
    last_helmet_frame = -1
    last_vest_frame   = -1
    last_helmet_ms: Optional[float] = None
    last_vest_ms:   Optional[float] = None

    last_helmet_by_id: Dict[int, Tuple[Optional[bool], int]] = {}
    last_vest_by_id:   Dict[int, Tuple[Optional[bool], int]] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx > max_frames:
            break

        # 1) 사람 트래킹(+키포인트)
        det = tracker.track_frame(frame)
        xyxys, scores, ids = det["xyxys"], det["scores"], det["ids"]
        kpt_xy, kpt_conf   = det["kpt_xy"], det["kpt_conf"]
        N = xyxys.shape[0]
        frame_has_detection = False

        # 2) 헬멧 (샘플링)
        helmet_boxes = None
        ran_helmet = (frame_idx % max(1, helmet_every) == 0)
        if ran_helmet:
            try:
                t0h = time.time()
                helmet_boxes = infer_helmet_boxes(
                    helmet_client, frame, helmet_label,
                    min_conf=h_min_conf, min_short_side=h_min_short_side
                )
                # 표준화: (xyxy, conf, class)
                helmet_boxes = _norm_triplets(helmet_boxes, default_cls="helmet")
                last_helmet_ms = (time.time() - t0h) * 1000.0
                last_helmet_frame = frame_idx
                cnt_h = 0 if helmet_boxes is None else len(helmet_boxes)
                print(f"[F{frame_idx}] helmet: n={cnt_h} {last_helmet_ms:.1f}ms")
            except Exception as e:
                print(f"[Frame {frame_idx}] Helmet API 실패: {type(e).__name__}: {e}")

        # 3) 조끼 (샘플링)
        kept_vests: List[Tuple[list, float, float]] = []
        ran_vest = (frame_idx % max(1, vest_every) == 0)
        dbg = {"raw":0,"geom":0,"color":0,"gate":0,"kept":0}
        if ran_vest:
            try:
                t0v = time.time()
                raw_vests = infer_vest_boxes(
                    vest_client, frame,
                    model_id=VEST_CONFIG["model_id"],
                    label_name=vest_label_for_call,
                    post_conf_thr=v_post_conf,
                    min_short_side=v_min_short_side
                )
                # 표준화: (xyxy, conf, class)
                raw_vests = _norm_triplets(raw_vests, default_cls="vest")

                last_vest_ms = (time.time() - t0v) * 1000.0
                last_vest_frame = frame_idx
                dbg["raw"] = len(raw_vests)

                person_boxes = [(xyxys[i].tolist(), float(scores[i])) for i in range(N)]
                H, W = frame.shape[:2]
                for (vxyxy, vconf, vcls) in raw_vests:
                    x1,y1,x2,y2 = map(int, vxyxy)
                    okg, _ = geom_ok(x1,y1,x2,y2, W,H,
                                     area_min_frac=v_area_min_frac,
                                     area_max_frac=v_area_max_frac,
                                     aspect_min=v_aspect_min,
                                     aspect_max=v_aspect_max)
                    if not okg:
                        continue
                    dbg["geom"] += 1

                    roi = frame[y1:y2, x1:x2]
                    ratio, _ = vest_color_ratio(roi)

                    # 하네스는 Hi-Vis 색상이 아닐 수 있음 → 색 비율 필터 우회
                    is_harness = ("harness" in _norm_label(vcls)) or (_norm_label(vcls) in HARNESS_ALIASES)
                    if (not is_harness) and (ratio < v_ratio_thr):
                        continue
                    dbg["color"] += 1

                    if not center_in_person_torso(x1,y1,x2,y2, person_boxes, v_torso_low, v_torso_high):
                        continue
                    dbg["gate"] += 1

                    kept_vests.append(([x1,y1,x2,y2], vconf, ratio))
                    dbg["kept"] += 1
            except Exception as e:
                print(f"[Frame {frame_idx}] Vest API 실패: {type(e).__name__}: {e}")

            print(f"[F{frame_idx}] vest raw={dbg['raw']} geom={dbg['geom']} color={dbg['color']} gate={dbg['gate']} kept={dbg['kept']}  {last_vest_ms:.1f}ms")

        # 4) 사람별 판정/그리기/로깅
        for i in range(N):
            this_kp_xy   = kpt_xy[i]   if (kpt_xy is not None and i < kpt_xy.shape[0])   else None
            this_kp_conf = kpt_conf[i] if (kpt_conf is not None and i < kpt_conf.shape[0]) else None
            this_id      = int(ids[i]) if (ids is not None and i < len(ids)) else None

            if person_visible(this_kp_xy, this_kp_conf, KPT_CONF):
                frame_has_detection = True
                part_text = get_person_part(this_kp_xy, this_kp_conf, KPT_CONF)

                # Helmet carry
                has_helmet = None
                if ran_helmet:
                    try:
                        if helmet_boxes and len(helmet_boxes) > 0:
                            has_helmet, _ = person_has_helmet(
                                xyxys[i].tolist(), helmet_boxes,
                                top_ratio=h_top_ratio, iou_thr=0.03
                            )
                        else:
                            has_helmet = None
                    except Exception as e:
                        print(f"[Frame {frame_idx}] Helmet match 실패: {type(e).__name__}: {e}")
                        has_helmet = None
                    if this_id is not None:
                        last_helmet_by_id[this_id] = (has_helmet, frame_idx)
                else:
                    if this_id is not None and this_id in last_helmet_by_id:
                        state, seen_at = last_helmet_by_id[this_id]
                        if (frame_idx - seen_at) <= h_carry_ttl:
                            has_helmet = state

                # Vest carry + 매칭
                has_vest = None
                vest_best_conf, vest_best_ratio = None, None
                if ran_vest:
                    if kept_vests:
                        v_found, v_conf, v_ratio = match_vest_to_person(
                            xyxys[i].tolist(), kept_vests,
                            torso_low=v_torso_low, torso_high=v_torso_high, iou_thr=0.05
                        )
                        if v_found:
                            has_vest = True
                            vest_best_conf, vest_best_ratio = v_conf, v_ratio
                        else:
                            has_vest = False  # 후보는 있었는데 내 몸통에 매칭 실패 → NO-VEST
                    else:
                        has_vest = False      # 프레임 내 조끼 후보 자체가 없으면 NO-VEST로 표기
                    if this_id is not None:
                        last_vest_by_id[this_id] = (has_vest, frame_idx)
                else:
                    if this_id is not None and this_id in last_vest_by_id:
                        state, seen_at = last_vest_by_id[this_id]
                        if (frame_idx - seen_at) <= v_carry_ttl:
                            has_vest = state

                # 시각화
                draw_person(
                    frame,
                    xyxys[i],
                    track_id=this_id,
                    kpt_xy=this_kp_xy,
                    kpt_conf=this_kp_conf,
                    helmet=has_helmet,
                    vest=has_vest,
                    part_text=part_text
                )

                # JSONL 로깅
                row = {
                    "video_path": os.path.abspath(video_path),
                    "frame_number": frame_idx,
                    "track_id": this_id,
                    "helmet": ("True" if has_helmet is True else "False" if has_helmet is False else "Skip"),
                    "vest":   ("True" if has_vest   is True else "False" if has_vest   is False else "Skip"),
                    "vest_conf": (None if vest_best_conf  is None else round(float(vest_best_conf), 6)),
                    "vest_color_ratio": (None if vest_best_ratio is None else round(float(vest_best_ratio), 6)),
                    "body_part": part_text
                }
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")

        # ---- 오버레이: procFPS + H/V Δ 및 지연(ms) ----
        t_now = time.time()
        inst = 1.0 / max(1e-6, (t_now - t_prev))
        ema = inst if ema is None else (0.9 * ema + 0.1 * inst)
        t_prev = t_now

        h_delta = (frame_idx - last_helmet_frame) if last_helmet_frame >= 0 else -1
        v_delta = (frame_idx - last_vest_frame)   if last_vest_frame   >= 0 else -1
        h_ms_txt = f"{last_helmet_ms:.0f}ms" if last_helmet_ms is not None else "--"
        v_ms_txt = f"{last_vest_ms:.0f}ms"   if last_vest_ms   is not None else "--"

        text = f"procFPS:{ema:.2f}  F:{frame_idx}  HΔ:{h_delta} H:{h_ms_txt}  VΔ:{v_delta} V:{v_ms_txt}"
        cv2.putText(frame, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2)

        if frame_has_detection:
            kept_people_frames += 1

        writer.write(frame)

    cap.release()
    writer.release()

    dt = time.time() - t0
    print(f"[DONE] {base}: frames={min(frame_idx, max_frames)}, kept_people_frames={kept_people_frames}, fps_proc={frame_idx/max(1e-6, dt):.2f}")
    print(f"[SAVE] video -> {out_path}")

# ------------------ 전체 실행 ------------------
def quick_e2e_all(max_frames_per_video: int = 120):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "video"), exist_ok=True)

    # 비디오 수집
    video_paths: List[str] = []
    for ext in ("*.mp4", "*.mov", "*.avi", "*.wmv", "*.mkv", "*.m4v"):
        video_paths += glob.glob(os.path.join(VIDEO_DIR, "**", ext), recursive=True)
    video_paths = sorted(set(video_paths))
    if not video_paths:
        raise RuntimeError(f"[ERR] VIDEO_DIR에 비디오 없음: {VIDEO_DIR}")

    print(f"[INFO] 총 {len(video_paths)}개 비디오 발견", flush=True)

    # YOLO + Tracker
    print("[INIT] YOLO 로드 중...", flush=True)
    model = YOLO(MODEL_PATH)
    tracker = Tracker(model=model, tracker_name=TRACKER_YAML, img_size=IMG_SIZE, det_conf=DET_CONF, device=DEVICE)
    print("[INIT] YOLO/Tracker 준비 완료", flush=True)

    # Helmet Client
    helmet_client = make_helmet_client(
        api_url=HELMET_CONFIG["api_url"],
        api_key=HELMET_CONFIG["api_key"],
        min_conf=float(HELMET_CONFIG.get("conf_thresh", 0.25)),
        iou_thresh=float(HELMET_CONFIG.get("iou_thresh", 0.5)),
        model_id=HELMET_CONFIG["model_id"]
    )
    helmet_every = int(HELMET_CONFIG.get("every", 15))

    # Vest Client
    vest_client = make_vest_client(
        api_url=VEST_CONFIG["api_url"],
        api_key=VEST_CONFIG["api_key"],
        conf_thresh=float(VEST_CONFIG.get("conf_thresh", 0.25)),
        iou_thresh=float(VEST_CONFIG.get("iou_thresh", 0.5)),
    )
    vest_every = int(VEST_CONFIG.get("every", 15))

    # 통합 JSONL
    jsonl_path = os.path.join(OUT_DIR, "quick_test_all.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for idx, vp in enumerate(video_paths, 1):
            print(f"\n[{idx}/{len(video_paths)}] 처리 시작: {vp}", flush=True)
            try:
                process_one_video(
                    video_path=vp, jf=jf, model=model, tracker=tracker,
                    helmet_client=helmet_client, vest_client=vest_client,
                    max_frames=max_frames_per_video,
                    helmet_every=helmet_every, vest_every=vest_every
                )
            except Exception as e:
                print(f"[ERROR] {vp}: {type(e).__name__}: {e}")
                continue

    print(f"\n[ALL DONE] JSONL -> {jsonl_path}")

if __name__ == "__main__":
    quick_e2e_all(max_frames_per_video=120)
