import os, json, glob, time, sys
import cv2
import numpy as np
from ultralytics import YOLO

from People_detection import ( person_visible, draw_person)
from tracker import Tracker

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import *

def process_video(video_path, tracker: Tracker, jsonl_file):
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

    frames_with_detection = 0
    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        det = tracker.track_frame(frame)
        xyxys, scores, ids = det["xyxys"], det["scores"], det["ids"]
        kpt_xy, kpt_conf   = det["kpt_xy"], det["kpt_conf"]

        frame_has_detection = False
        N = xyxys.shape[0]

        for i in range(N):
            this_kp_xy   = kpt_xy[i]   if kpt_xy   is not None and i < kpt_xy.shape[0]   else None
            this_kp_conf = kpt_conf[i] if kpt_conf is not None and i < kpt_conf.shape[0] else None
            this_id      = int(ids[i]) if ids is not None and i < len(ids)              else None

            if person_visible(this_kp_xy, this_kp_conf, KPT_CONF):
                frame_has_detection = True
                draw_person(
                    frame,
                    xyxys[i],
                    this_id,
                    this_kp_xy,
                    this_kp_conf
                )

                row = {
                    "video_path": os.path.abspath(video_path),
                    "frame_number": frame_idx,
                    "track_id": this_id,
                    "bbox_xyxy": [float(x) for x in xyxys[i].tolist()],
                    "score": float(scores[i]),
                    "kpts_xy": None if this_kp_xy is None else [[float(a), float(b)] for a,b in this_kp_xy.tolist()],
                    "kpts_conf": None if this_kp_conf is None else [
                        None if c is None else float(c) for c in this_kp_conf.tolist()
                    ],
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

    # 비디오 수집
    video_paths = []
    for ext in ("*.mp4", "*.mov", "*.avi", "*.wmv", "*.mkv"):
        video_paths.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
    video_paths.sort()


    print(f"찾은 동영상 파일: {len(video_paths)}개")

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for vp in video_paths:
            process_video(vp, tracker, jf)

    print(f"JSONL 파일: {jsonl_path}")

if __name__ == "__main__":
    main()
