import os, glob, time, sys, csv
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# PROJECT_ROOT 설정은 그대로 유지합니다.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) 

from Model.src.yolo.people_detection import *
from Model.src.yolo.ppe_detection import PPEDetector
from Model.src.yolo.tracker import Tracker
from config import *

VIDEO_DIR =  "../../Data/Real"
OUT_DIR = "../../Data/output/compare"
MODEL_PPE_PATH = os.path.join(PROJECT_ROOT, "Model", "checkpoints", "yolo11n_safety.pt")

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


def process_video(video_path: str, model_instance: YOLO):
    """
    하나의 비디오 파일에 대해 YOLO 모델 성능을 측정합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: '{video_path}' 파일을 열 수 없습니다.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    valid_frames = 0
    start_time = time.perf_counter()

    print(f"처리 중: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델 감지 실행
        results = model_instance(
            frame,
            imgsz=IMG_SIZE,
            conf=DET_CONF,
            device=DEVICE,
            verbose=False,
        )

        # 감지된 객체(사람)가 있으면 유효 프레임으로 카운트
        if len(results[0].boxes) > 0:
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
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # 비디오 수집
    video_paths = []
    for ext in ("*.mp4", "*.mov", "*.avi", "*.wmv", "*.mkv"):
        video_paths.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
    video_paths.sort()

    if not video_paths:
        print(f"오류: '{VIDEO_DIR}' 경로에 비디오 파일이 없습니다.")
        return

    # --- 1. PeopleDetector (YOLO) 성능 측정 ---
    print("PeopleDetector 성능 측정 시작...")
    # people_detector = YOLO(MODEL_PEOPLE_PATH)
    people_detector = YOLO(MODEL_PATH)
    for video_path in video_paths:
        summary = process_video(video_path, people_detector)
        if summary:
            append_csv_row(csv_path, {
                "total_frames": summary["total_frames"],
                "valid_frames": summary["valid_frames"],
                "processed_fps": summary["processed_fps"],
                # "model_path": os.path.abspath(MODEL_PEOPLE_PATH),
                "model_path": os.path.abspath(MODEL_PATH),
                "tracker_yaml": TRACKER_YAML,
                "img_size": IMG_SIZE,
                "det_conf": DET_CONF,
                "device": DEVICE,
            })
    print("PeopleDetector 성능 측정 완료.")

    # --- 2. PPEDetector (YOLO) 성능 측정 ---
    print("PPEDetector 성능 측정 시작...")
    ppe_detector = YOLO(MODEL_PPE_PATH)
    for video_path in video_paths:
        summary = process_video(video_path, ppe_detector)
        if summary:
            append_csv_row(csv_path, {
                "total_frames": summary["total_frames"],
                "valid_frames": summary["valid_frames"],
                "processed_fps": summary["processed_fps"],
                "model_path": os.path.abspath(MODEL_PPE_PATH),
                "tracker_yaml": TRACKER_YAML,
                "img_size": IMG_SIZE,
                "det_conf": DET_CONF,
                "device": DEVICE,
            })
    print("PPEDetector 성능 측정 완료.")

    print(f"\n모든 성능 결과가 {csv_path}에 저장되었습니다. 🎉")

if __name__ == "__main__":
    main()