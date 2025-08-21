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
    DEVICE, KPT_CONF, PPE_CONFIG
)

# --- 외부 모듈 ---
from tracker import Tracker
from People_detection import get_person_part, person_visible, draw_person

# 단일 API 모듈 (헬멧+조끼 한 번에)
# 경로 차이를 위한 안전한 import
try:
    from model.src.ppe_detection import (
        make_ppe_client, infer_ppe, person_has_helmet,
        vest_color_ratio, geom_ok, center_in_person_torso, match_vest_to_person
    )
except ImportError:
    from ppe_detection import (
        make_ppe_client, infer_ppe, person_has_helmet,
        vest_color_ratio, geom_ok, center_in_person_torso, match_vest_to_person
    )

# ------------------ 한 비디오 처리 ------------------
def process_one_video(video_path: str,
                      jf,
                      model: YOLO,
                      tracker: "Tracker",
                      ppe_client,
                      max_frames: int,
                      ppe_every: int):
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

    # --- PPE 파라미터 (config에서 읽음) ---
    min_short_side  = int(PPE_CONFIG.get("min_short_side", 640))
    post_conf_thr   = float(PPE_CONFIG.get("post_conf_thr", 0.35))
    carry_ttl       = int(PPE_CONFIG.get("carry_ttl", 90))
    top_ratio       = float(PPE_CONFIG.get("helmet_top_ratio", 0.60))

    # Vest 후처리(색/형상/몸통위치 필터)
    v_ratio_thr     = float(PPE_CONFIG.get("vest_ratio_thr", 0.008))
    v_area_min_frac = float(PPE_CONFIG.get("vest_area_min_frac", 0.0015))
    v_area_max_frac = float(PPE_CONFIG.get("vest_area_max_frac", 0.25))
    v_aspect_min    = float(PPE_CONFIG.get("vest_aspect_min", 0.35))
    v_aspect_max    = float(PPE_CONFIG.get("vest_aspect_max", 3.0))
    v_torso_low     = float(PPE_CONFIG.get("vest_torso_low", 0.15))
    v_torso_high    = float(PPE_CONFIG.get("vest_torso_high", 0.90))

    # 선택: 색모드 키가 있는 경우만 사용(없으면 함수 기본값 사용)
    v_color_mode    = str(PPE_CONFIG.get("vest_color_mode", "hivis")).lower()

    # --- FPS 오버레이 옵션 ---
    show_fps    = bool(PPE_CONFIG.get("show_fps", True))          # 없으면 기본 True
    fps_alpha   = float(PPE_CONFIG.get("fps_ema_alpha", 0.10))    # EMA 스무딩 계수
    ema_proc_fps = None
    last_tick    = time.time()

    frame_idx = 0
    kept_people_frames = 0
    t0 = time.time()

    # track_id 기준 상태 보간
    last_helmet_by_id: Dict[int, Tuple[Optional[bool], int]] = {}
    last_vest_by_id:   Dict[int, Tuple[Optional[bool], int]] = {}

    # 샘플링 캐시 (ppe_every 프레임마다 갱신)
    cached_helmets: List[Tuple[list, float, str]] = []
    cached_vests:   List[Tuple[list, float, str]] = []

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

        # 2) 단일 PPE API 호출 (샘플링)
        if frame_idx % max(1, ppe_every) == 0:
            try:
                p = infer_ppe(
                    ppe_client, frame,
                    model_id=PPE_CONFIG["model_id"],
                    post_conf_thr=post_conf_thr,
                    min_short_side=min_short_side
                )
                cached_helmets = p.get("helmets", [])
                cached_vests   = p.get("vests", [])
                print(f"[F{frame_idx}] PPE raw: helmets={len(cached_helmets)} vests={len(cached_vests)}")
            except Exception as e:
                print(f"[Frame {frame_idx}] PPE API 실패: {type(e).__name__}: {e}")

        # 3) 조끼 후보 후처리(색/형상/몸통 위치) → kept_vests
        kept_vests: List[Tuple[list, float, float]] = []
        if cached_vests:
            person_boxes = [(xyxys[i].tolist(), float(scores[i])) for i in range(N)]
            H, W = frame.shape[:2]
            for (vxyxy, vconf, vclass) in cached_vests:
                x1, y1, x2, y2 = map(int, vxyxy)
                okg, _ = geom_ok(x1, y1, x2, y2, W, H,
                                 area_min_frac=v_area_min_frac,
                                 area_max_frac=v_area_max_frac,
                                 aspect_min=v_aspect_min,
                                 aspect_max=v_aspect_max)
                if not okg:
                    continue

                roi = frame[y1:y2, x1:x2]
                # vest_color_ratio 시그니처(모드 지원/미지원) 모두 호환
                try:
                    ratio, _ = vest_color_ratio(roi, mode=v_color_mode)
                except TypeError:
                    ratio, _ = vest_color_ratio(roi)

                if ratio < v_ratio_thr:
                    continue
                if not center_in_person_torso(x1, y1, x2, y2, person_boxes, v_torso_low, v_torso_high):
                    continue
                kept_vests.append(([x1, y1, x2, y2], vconf, ratio))
        if frame_idx % max(1, ppe_every) == 0:
            print(f"[F{frame_idx}] vest kept after filters={len(kept_vests)}")

        # 4) 사람별 판정/그리기/로깅
        for i in range(N):
            this_kp_xy   = kpt_xy[i]   if (kpt_xy is not None and i < kpt_xy.shape[0])   else None
            this_kp_conf = kpt_conf[i] if (kpt_conf is not None and i < kpt_conf.shape[0]) else None
            this_id      = int(ids[i]) if (ids is not None and i < len(ids)) else None

            if person_visible(this_kp_xy, this_kp_conf, KPT_CONF):
                frame_has_detection = True
                part_text = get_person_part(this_kp_xy, this_kp_conf, KPT_CONF)

                # Helmet carry/보간
                has_helmet: Optional[bool] = None
                if cached_helmets:
                    try:
                        # (신규 서명) top_ratio, iou_thr, kpt 사용
                        has_helmet, _ = person_has_helmet(
                            xyxys[i].tolist(), cached_helmets,
                            top_ratio=top_ratio, iou_thr=0.04,
                            kpt_xy=this_kp_xy, kpt_conf=this_kp_conf, kpt_thr=KPT_CONF
                        )
                    except TypeError:
                        # (구 서명) top_ratio, iou_thr만 있는 경우
                        has_helmet, _ = person_has_helmet(
                            xyxys[i].tolist(), cached_helmets,
                            top_ratio=top_ratio, iou_thr=0.04
                        )
                    except Exception as e:
                        print(f"[Frame {frame_idx}] Helmet match 실패: {type(e).__name__}: {e}")
                        has_helmet = None
                    if this_id is not None:
                        last_helmet_by_id[this_id] = (has_helmet, frame_idx)
                else:
                    if this_id is not None and this_id in last_helmet_by_id:
                        state, seen_at = last_helmet_by_id[this_id]
                        if (frame_idx - seen_at) <= carry_ttl:
                            has_helmet = state

                # Vest carry/보간 + 매칭
                has_vest: Optional[bool] = None
                vest_best_conf, vest_best_ratio = None, None
                if kept_vests:
                    v_found, v_conf, v_ratio = match_vest_to_person(
                        xyxys[i].tolist(), kept_vests,
                        torso_low=v_torso_low, torso_high=v_torso_high, iou_thr=0.05
                    )
                    if v_found:
                        has_vest = True
                        vest_best_conf, vest_best_ratio = v_conf, v_ratio
                    else:
                        has_vest = False
                    if this_id is not None:
                        last_vest_by_id[this_id] = (has_vest, frame_idx)
                else:
                    if this_id is not None and this_id in last_vest_by_id:
                        state, seen_at = last_vest_by_id[this_id]
                        if (frame_idx - seen_at) <= carry_ttl:
                            has_vest = state

                # 시각화 (violation-first 색 규칙)
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

                # JSONL
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

        # --- FPS 오버레이 (프레임 끝에서 갱신해 그리고 저장) ---
        now = time.time()
        inst_proc_fps = 1.0 / max(1e-6, (now - last_tick))
        last_tick = now
        if ema_proc_fps is None:
            ema_proc_fps = inst_proc_fps
        else:
            ema_proc_fps = (1.0 - fps_alpha) * ema_proc_fps + fps_alpha * inst_proc_fps

        if show_fps and ema_proc_fps is not None:
            label = f"srcFPS:{fps:.1f}  procFPS:{ema_proc_fps:.2f}  F:{frame_idx}"
            # 테두리(가독성)
            cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        if frame_has_detection:
            kept_people_frames += 1

    cap.release()
    writer.release()

    dt = time.time() - t0
    fps_proc = frame_idx / max(1e-6, dt)
    print(f"[DONE] {base}: frames={min(frame_idx, max_frames)}, kept_people_frames={kept_people_frames}, fps_proc={fps_proc:.2f}")
    print(f"[SAVE] video -> {out_path}]")

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

    # 단일 PPE Client
    ppe_client = make_ppe_client(
        api_url=PPE_CONFIG["api_url"],
        api_key=PPE_CONFIG["api_key"],
        conf_thresh=float(PPE_CONFIG.get("conf_thresh", 0.25)),
        iou_thresh=float(PPE_CONFIG.get("iou_thresh", 0.5)),
        model_id=PPE_CONFIG["model_id"]
    )
    ppe_every = int(PPE_CONFIG.get("every", 15))

    # 통합 JSONL
    jsonl_path = os.path.join(OUT_DIR, "quick_test_all.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for idx, vp in enumerate(video_paths, 1):
            print(f"\n[{idx}/{len(video_paths)}] 처리 시작: {vp}", flush=True)
            try:
                process_one_video(
                    video_path=vp, jf=jf, model=model, tracker=tracker,
                    ppe_client=ppe_client,
                    max_frames=max_frames_per_video,
                    ppe_every=ppe_every
                )
            except Exception as e:
                print(f"[ERROR] {vp}: {type(e).__name__}: {e}")
                continue

    print(f"\n[ALL DONE] JSONL -> {jsonl_path}")

if __name__ == "__main__":
    quick_e2e_all(max_frames_per_video=120)
