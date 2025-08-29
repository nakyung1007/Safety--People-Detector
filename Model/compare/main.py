import os, glob, time, sys, csv
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# PROJECT_ROOT 설정은 그대로 유지합니다.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) 

from Model.src.yolo.tracker import Tracker
from config import *

VIDEO_DIR = "../../Data/Real"
OUT_DIR = "../../Data/output/compare"

CSV_FIELDS = [
    "total_frames", "valid_frames", "processed_fps",
    "model_path", "tracker_yaml",
    "img_size", "det_conf", "device"
]

def append_csv_row(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if need_header:
            w.writeheader()
        safe_row = {k: row.get(k, "") for k in CSV_FIELDS}
        w.writerow(safe_row)

def process_video_tracker(video_path: str, tracker_instance: Tracker, model_name: str, tracker_name: str):
    """
    하나의 비디오 파일에 대해 Tracker 성능을 측정합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: '{video_path}' 파일을 열 수 없습니다.")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    valid_frames = 0
    
    print(f"처리 중: {video_path} | 모델: {model_name}, 트래커: {tracker_name}")
    
    start_time = time.perf_counter()
    
    # 트래커의 track_video_stream 메소드를 사용하고 제너레이터를 순회
    results = tracker_instance.track_video_stream(video_path)
    
    # 제너레이터는 반복문을 통해 결과를 하나씩 가져옵니다.
    for r in results:
        # 이 루프를 한 번 돌 때마다 한 프레임씩 처리됩니다.
        if r.boxes.id is not None and len(r.boxes.id) > 0:
            valid_frames += 1
            
    end_time = time.perf_counter()
    duration = end_time - start_time
    processed_fps = total_frames / duration if duration > 0 else 0.0

    cap.release()
    
    print(f"완료: {video_path} | 유효프레임: {valid_frames}/{total_frames} | {processed_fps:.2f} FPS(처리)")
    
    return {
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "processed_fps": round(processed_fps, 2),
    }

# ---
# main 함수
# ---
def main():
    
    csv_path = os.path.join(OUT_DIR, "performance_results.csv")

    # 비디오 수집
    video_paths = []
    for ext in ("*.mp4", "*.mov", "*.avi", "*.wmv", "*.mkv"):
        video_paths.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
    video_paths.sort()

    if not video_paths:
        print(f"오류: '{VIDEO_DIR}' 경로에 비디오 파일이 없습니다.")
        return

    # config.py에 정의된 값 사용
    # model_path = os.path.join(PROJECT_ROOT, "Model", MODEL_PEOPLE_PATH) # yolo11n-pose.pt
    model_path = os.path.join(PROJECT_ROOT, "Model", MODEL_PATH) # yolo8n-pose.pt
    tracker_yaml = TRACKER_YAML
    # model_name = os.path.basename(MODEL_PEOPLE_PATH) # yolo11n-pose.pt
    model_name = os.path.basename(MODEL_PATH) # yolo8n-pose.pt

    print(f"\n--- 성능 측정 시작: 모델={model_name}, 트래커={tracker_yaml} ---")
    
    # YOLO 모델 로드
    model_instance = YOLO(model_path)
    
    # Tracker 인스턴스화
    tracker_instance = Tracker(
        model=model_instance,
        tracker_name=tracker_yaml
    )

    for video_path in video_paths:
        summary = process_video_tracker(video_path, tracker_instance, model_name, tracker_yaml)
        if summary:
            append_csv_row(csv_path, {
                "total_frames": summary["total_frames"],
                "valid_frames": summary["valid_frames"],
                "processed_fps": summary["processed_fps"],
                "model_path": model_name,
                "tracker_yaml": tracker_yaml,
                "img_size": IMG_SIZE,
                "det_conf": DET_CONF,
                "device": DEVICE,
            })
    print(f"--- 성능 측정 완료: 모델={model_name}, 트래커={tracker_yaml} ---")

    print(f"\n모든 성능 결과가 {csv_path}에 저장되었습니다. 🎉")

if __name__ == "__main__":
    main()