import os, json, glob, time, sys
import cv2
import numpy as np
from ultralytics import YOLO

from People_detection import *
from tracker import *

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import *

from helmet_Detection import (
    make_helmet_client, infer_helmet_boxes, person_has_helmet
)

def process_video(video_path, tracker: Tracker, jsonl_file,
                  helmet_client, helmet_label_name: str,
                  helmet_conf: float = 0.65, helmet_min_ss: int = 640, helmet_every: int=15, carry_ttl: int = 60):
    cap = cv2.VideoCapture(video_path)

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(OUT_DIR, "video", f"{base}_tracked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"처리 중: {video_path}")
    print(f"총 프레임: {total}, FPS: {fps}")
    
    last_helmet_by_id = {}

    frames_with_detection = 0
    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        #1) 사람 트래킹
        det = tracker.track_frame(frame)
        xyxys, scores, ids = det["xyxys"], det["scores"], det["ids"]
        kpt_xy, kpt_conf   = det["kpt_xy"], det["kpt_conf"]

        frame_has_detection = False
        N = xyxys.shape[0]
        
        #2) helmet Detection
       # 2) Helmet Detection
        want_infer = (frame_idx % helmet_every == 0)
        ran_infer  = False
        helmet_boxes = None

        if want_infer:
            try:
                helmet_boxes = infer_helmet_boxes(
                    helmet_client, frame, helmet_label_name,
                    min_conf=helmet_conf, min_short_side=helmet_min_ss
                )
                ran_infer = True
                print(f"[Frame {frame_idx}] Helmet detection 실행 → 결과 {len(helmet_boxes)}개")
            except Exception as e:
                # ★ 여기서 절대 크래시하지 않음: carry 로 대체
                print(f"[Frame {frame_idx}] Helmet detection 실패: {type(e).__name__}: {e}")
                helmet_boxes = None  # 아래에서 carry 사용
        else:
            print(f"[Frame {frame_idx}] Helmet detection SKIP (carry 사용)")


        for i in range(N):
            this_kp_xy   = kpt_xy[i]   if kpt_xy   is not None and i < kpt_xy.shape[0]   else None
            this_kp_conf = kpt_conf[i] if kpt_conf is not None and i < kpt_conf.shape[0] else None
            this_id      = int(ids[i]) if ids is not None and i < len(ids)              else None

            if person_visible(this_kp_xy, this_kp_conf, KPT_CONF):
                frame_has_detection = True
                part_text = get_person_part(this_kp_xy, this_kp_conf, KPT_CONF)

                # 3) 매칭 or carry
                has_helmet = None
                if ran_infer:
                    # 새로 감지 → 캐시에 저장
                    has_helmet, helmet_score = person_has_helmet(
                        xyxys[i].tolist(), helmet_boxes, top_ratio=0.45, iou_thr=0.05
                    )
                    if this_id is not None:
                        last_helmet_by_id[this_id] = (has_helmet, frame_idx)
                else:
                    # 감지 스킵/실패 → 최근 상태 carry (TTL 내)
                    if this_id is not None and this_id in last_helmet_by_id:
                        state, seen_at = last_helmet_by_id[this_id]
                        if (carry_ttl is None) or ((frame_idx - seen_at) <= carry_ttl):
                            has_helmet = state  # ← 유지

                draw_person(
                    frame, xyxys[i], this_id,
                    this_kp_xy, this_kp_conf,
                    helmet=has_helmet
                )
                
                # JSONL 기록: 유지된 값이 있으면 True/False로 기록, 없으면 "Skip"
                helmet_json = (
                    "True" if has_helmet is True else
                    "False" if has_helmet is False else
                    "Skip"
                )

                row = {
                    "video_path": os.path.abspath(video_path),
                    "frame_number": frame_idx,
                    "track_id": this_id,
                    "helmet": helmet_json,   # "True" or "False" or "Skip"
                    "body part": part_text
                }
                jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

        if frame_has_detection:
            frames_with_detection += 1

        writer.write(frame)

        if total > 0 and frame_idx % 30 == 0:
            print(f"진행률: {frame_idx/total*100:.1f}% ({frame_idx}/{total})")

    cap.release()
    writer.release()

    dt = time.time() - t0

    print(f"완료: {video_path}  |  유효프레임: {frames_with_detection}/{frame_idx}  |  저장: {out_path}  |  {frame_idx/dt:.2f} FPS(처리)")

def main():
    jsonl_path = os.path.join(OUT_DIR, "detections.jsonl")

    # 모델 로드
    model = YOLO(MODEL_PATH)  

    # 트래커 래퍼
    tracker = Tracker(
        model=model,
        tracker_name=TRACKER_YAML,  
        img_size=IMG_SIZE,
        det_conf=DET_CONF,
        device=DEVICE
    )

    # 헬멧
    helmet_min_conf = 0.55
    helmet_client = make_helmet_client(
        api_url=HELMET_CONFIG["api_url"],
        api_key=HELMET_CONFIG["api_key"],
        min_conf=helmet_min_conf,
        iou_thresh=HELMET_CONFIG["iou_thresh"],
        model_id=HELMET_CONFIG["model_id"]
    )
    helmet_label_name = str(HELMET_CONFIG["label_name"]).strip()

    # 비디오 수집
    video_paths = []
    for ext in ("*.mp4", "*.mov", "*.avi", "*.wmv", "*.mkv"):
        video_paths.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
    video_paths.sort()

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for vp in video_paths:
            process_video(
                vp, tracker, jf,
                helmet_client=helmet_client,
                helmet_label_name=helmet_label_name,
                helmet_conf=helmet_min_conf,
                helmet_min_ss=640,
            )
            
    print(f"JSONL 파일: {jsonl_path}")

if __name__ == "__main__":
    main()
