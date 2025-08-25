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

# 단일 API 모듈 (헬멧+조끼 한 번에) – 경로 차이 안전 임포트
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


# ======================= Track-level PPE History =======================
class PPEHistory:
    """
    트랙별 상태 히스테리시스:
      - 같은 관측(True/False)이 연속 k프레임 누적되면 상태 확정
      - 관측 None이면 상태 유지
      - 마지막 갱신 후 ttl 프레임이 지나면 상태 무효화(None)
    """
    def __init__(self, ttl: int = 90, pos_req: int = 2, neg_req: int = 2):
        self.ttl = int(ttl)
        self.pos_req = int(pos_req)
        self.neg_req = int(neg_req)
        # key=(track_id, kind['helmet'|'vest']) -> dict
        self.mem: Dict[Tuple[int, str], Dict[str, float]] = {}

    def observe(self, track_id: Optional[int], kind: str, obs: Optional[bool], frame_idx: int):
        if track_id is None:
            return
        key = (track_id, kind)
        d = self.mem.get(key, {"state": None, "pos": 0, "neg": 0, "updated": -1})
        if obs is True:
            d["pos"] = d.get("pos", 0) + 1
            d["neg"] = 0
            if d["pos"] >= self.pos_req:
                d["state"] = True
                d["updated"] = frame_idx
        elif obs is False:
            d["neg"] = d.get("neg", 0) + 1
            d["pos"] = 0
            if d["neg"] >= self.neg_req:
                d["state"] = False
                d["updated"] = frame_idx
        # None이면 streak 유지
        self.mem[key] = d

    def get(self, track_id: Optional[int], kind: str, frame_idx: int) -> Optional[bool]:
        if track_id is None:
            return None
        d = self.mem.get((track_id, kind))
        if not d or d["updated"] < 0:
            return None
        if (frame_idx - d["updated"]) > self.ttl:
            return None
        return d["state"]


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
    min_short_side  = int(PPE_CONFIG.get("min_short_side", 640))     # 서버로 보낼 최소 짧변
    post_conf_thr   = float(PPE_CONFIG.get("post_conf_thr", 0.35))   # 클라 confidence 필터
    carry_ttl       = int(PPE_CONFIG.get("carry_ttl", 90))           # 상태 TTL(히스테리시스에도 사용)
    top_ratio       = float(PPE_CONFIG.get("helmet_top_ratio", 0.60))
    helmet_cover    = float(PPE_CONFIG.get("helmet_cover_frac", 0.35))
    eye_margin      = float(PPE_CONFIG.get("helmet_eye_margin", 0.04))

    # Vest 후처리(색/형상/몸통위치 필터)
    v_ratio_thr     = float(PPE_CONFIG.get("vest_ratio_thr", 0.008))
    v_area_min_frac = float(PPE_CONFIG.get("vest_area_min_frac", 0.0015))
    v_area_max_frac = float(PPE_CONFIG.get("vest_area_max_frac", 0.25))
    v_aspect_min    = float(PPE_CONFIG.get("vest_aspect_min", 0.35))
    v_aspect_max    = float(PPE_CONFIG.get("vest_aspect_max", 3.0))
    v_torso_low     = float(PPE_CONFIG.get("vest_torso_low", 0.15))
    v_torso_high    = float(PPE_CONFIG.get("vest_torso_high", 0.90))
    v_color_mode    = str(PPE_CONFIG.get("vest_color_mode", "hivis")).lower()

    # 조건부 호출용 파라미터
    min_person_h_px = int(PPE_CONFIG.get("min_person_h", 80))  # 너무 작은 사람만 있으면 PPE 스킵
    pos_req         = int(PPE_CONFIG.get("hysteresis_pos", 2))
    neg_req         = int(PPE_CONFIG.get("hysteresis_neg", 2))

    # FPS(EMA)
    ema = None
    t_prev = time.time()

    # PPE 히스토리
    hist = PPEHistory(ttl=carry_ttl, pos_req=pos_req, neg_req=neg_req)

    # PPE 캐시(동기 호출 결과)
    cached_helmets: List[Tuple[list, float, str]] = []
    cached_vests:   List[Tuple[list, float, str]] = []
    last_ppe_frame: int = -1
    last_ppe_ms: Optional[float] = None

    # 트랙 변화 감지용
    prev_ids: set = set()

    frame_idx = 0
    kept_people_frames = 0
    t0_all = time.time()

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
        N = 0 if xyxys is None else xyxys.shape[0]
        frame_has_detection = False

        # 현재 트랙 id 집합
        cur_ids = set()
        if ids is not None:
            for i in range(len(ids)):
                try:
                    cur_ids.add(int(ids[i]))
                except Exception:
                    pass

        # 2) PPE 호출 필요 여부 판단(동기)
        force_by_track = (frame_idx == 1) or (cur_ids != prev_ids)  # 새 트랙 등장/소실
        periodic = (frame_idx % max(1, ppe_every) == 0)

        # 사람 조건 체크(없거나 너무 작은 경우 PPE 스킵)
        call_allowed = False
        if xyxys is not None and xyxys.shape[0] > 0:
            max_h = 0
            for i in range(xyxys.shape[0]):
                x1, y1, x2, y2 = map(float, xyxys[i].tolist())
                max_h = max(max_h, int(y2 - y1))
            call_allowed = (max_h >= min_person_h_px)

        do_call = call_allowed and (force_by_track or periodic)

        if do_call:
            t0 = time.time()
            try:
                p = infer_ppe(
                    ppe_client, frame,
                    model_id=PPE_CONFIG["model_id"],
                    post_conf_thr=post_conf_thr,
                    min_short_side=min_short_side
                )
                cached_helmets = p.get("helmets", [])
                cached_vests   = p.get("vests", [])
                last_ppe_frame = frame_idx
                last_ppe_ms = (time.time() - t0) * 1000.0
                print(f"[F{frame_idx}] PPE call -> helmets={len(cached_helmets)} vests={len(cached_vests)} ({last_ppe_ms:.1f}ms)")
            except Exception as e:
                print(f"[F{frame_idx}] PPE infer FAIL: {type(e).__name__}: {e}")

        # 3) 조끼 후보 후처리(색/형상/몸통 위치) → kept_vests
        kept_vests_for_match: List[Tuple[list, float, float]] = []
        if cached_vests and xyxys is not None and xyxys.shape[0] > 0:
            person_boxes = [(xyxys[i].tolist(), float(scores[i])) for i in range(xyxys.shape[0])]
            H, W = frame.shape[:2]
            for (vxyxy, vconf, _c) in cached_vests:
                x1, y1, x2, y2 = map(int, vxyxy)
                okg, _ = geom_ok(x1, y1, x2, y2, W, H,
                                 area_min_frac=v_area_min_frac,
                                 area_max_frac=v_area_max_frac,
                                 aspect_min=v_aspect_min,
                                 aspect_max=v_aspect_max)
                if not okg:
                    continue
                roi = frame[y1:y2, x1:x2]
                # vest_color_ratio(모드 인자 지원/미지원 모두 호환)
                try:
                    ratio, _ = vest_color_ratio(roi, mode=v_color_mode)
                except TypeError:
                    ratio, _ = vest_color_ratio(roi)
                if ratio < v_ratio_thr:
                    continue
                if not center_in_person_torso(x1, y1, x2, y2, person_boxes, v_torso_low, v_torso_high):
                    continue
                kept_vests_for_match.append(([x1, y1, x2, y2], vconf, ratio))

        if do_call:
            print(f"[F{frame_idx}] vest kept after filters={len(kept_vests_for_match)}")

        # 4) 사람별 판정/그리기/로깅
        if xyxys is not None:
            for i in range(xyxys.shape[0]):
                this_kp_xy   = kpt_xy[i]   if (kpt_xy is not None and i < kpt_xy.shape[0])   else None
                this_kp_conf = kpt_conf[i] if (kpt_conf is not None and i < kpt_conf.shape[0]) else None
                this_id      = int(ids[i]) if (ids is not None and i < len(ids)) else None

                if person_visible(this_kp_xy, this_kp_conf, KPT_CONF):
                    frame_has_detection = True
                    part_text = get_person_part(this_kp_xy, this_kp_conf, KPT_CONF)

                    # --- 한 프레임의 관측값(즉시) ---
                    inst_helmet: Optional[bool] = None
                    if cached_helmets:
                        try:
                            inst_helmet, _ = person_has_helmet(
                                xyxys[i].tolist(), cached_helmets,
                                top_ratio=top_ratio, iou_thr=0.04,
                                min_cover_frac=helmet_cover, eye_margin_frac=eye_margin,
                                kpt_xy=this_kp_xy, kpt_conf=this_kp_conf, kpt_thr=KPT_CONF
                            )
                        except TypeError:
                            # 구 시그니처 호환
                            inst_helmet, _ = person_has_helmet(
                                xyxys[i].tolist(), cached_helmets,
                                top_ratio=top_ratio, iou_thr=0.04
                            )

                    inst_vest: Optional[bool] = None
                    vest_best_conf, vest_best_ratio = None, None
                    if kept_vests_for_match:
                        v_found, v_conf, v_ratio = match_vest_to_person(
                            xyxys[i].tolist(), kept_vests_for_match,
                            torso_low=v_torso_low, torso_high=v_torso_high, iou_thr=0.05
                        )
                        if v_found:
                            inst_vest = True
                            vest_best_conf, vest_best_ratio = v_conf, v_ratio
                        else:
                            inst_vest = False

                    # --- 히스토리 적용(안정화) ---
                    hist.observe(this_id, "helmet", inst_helmet, frame_idx)
                    hist.observe(this_id, "vest",   inst_vest,   frame_idx)
                    has_helmet = hist.get(this_id, "helmet", frame_idx)
                    has_vest   = hist.get(this_id, "vest",   frame_idx)

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

        # --- 오버레이: procFPS & PPEΔ & PPEms ---
        t_now = time.time()
        inst = 1.0 / max(1e-6, (t_now - t_prev))
        ema = inst if ema is None else (0.9 * ema + 0.1 * inst)
        t_prev = t_now

        ppe_delta = (frame_idx - last_ppe_frame) if last_ppe_frame >= 0 else -1
        ppe_ms_txt = f"{last_ppe_ms:.0f}ms" if last_ppe_ms is not None else "--"
        cv2.putText(frame, f"procFPS:{ema:.2f}  F:{frame_idx}  PPEΔ:{ppe_delta}  PPE:{ppe_ms_txt}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2)

        writer.write(frame)
        if frame_has_detection:
            kept_people_frames += 1

        prev_ids = cur_ids

    cap.release()
    writer.release()

    dt = time.time() - t0_all
    fps_proc = frame_idx / max(1e-6, dt)
    print(f"[DONE] {os.path.basename(video_path)}: frames={min(frame_idx, max_frames)}, kept_people_frames={kept_people_frames}, fps_proc={fps_proc:.2f}")
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

    # 단일 PPE Client (동기 호출)
    ppe_client = make_ppe_client(
        api_url=PPE_CONFIG["api_url"],
        api_key=PPE_CONFIG["api_key"],
        conf_thresh=float(PPE_CONFIG.get("conf_thresh", 0.30)),
        iou_thresh=float(PPE_CONFIG.get("iou_thresh", 0.5)),
        model_id=PPE_CONFIG["model_id"]
    )
    ppe_every = int(PPE_CONFIG.get("every", 30))
    print(f"[CONFIG] ppe_every={ppe_every}, min_short_side={int(PPE_CONFIG.get('min_short_side', 640))}, vest_color_mode={str(PPE_CONFIG.get('vest_color_mode', 'hivis')).lower()}")

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
