# -*- coding: utf-8 -*-
"""
VIDEO_DIR 안의 모든 동영상을 처리:
1) 사람 추적(yolo11n-pose + ByteTrack/BOT-SORT)
2) PPE 탐지(yolo11n-safety: hat/nohat/vest/novest)
3) 매칭(중심점 포함 or IoU>=0.05)
4) 결과:
   - 주석(바운딩박스/텍스트) 오버레이된 동영상: OUT_DIR/videos/<원본>_safety.mp4
   - bbox_id 별 크롭 이미지: OUT_DIR/crops/<원본_stem>/id_<ID>/<frame>.jpg
   - 프레임별 로그 CSV: OUT_DIR/logs/<원본_stem>_tracks.csv

필요 모듈:
- tracker.py : Tracker 클래스를 사용
- ppe_detection.py : PPEDetector, assign_ppe_to_person_boxes
- people_detection.py : draw_person, get_person_part
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import csv
import cv2
import numpy as np
from ultralytics import YOLO
import os, sys
import time
import logging
import json
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 프로젝트 모듈
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import *

from tracker import *
from ppe_detection import *
from people_detection import *

# ---------------- 경로 유틸 ----------------
def _resolve(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent / p).resolve()

def _resolve_ckpt(path_str: str) -> Path:
    """
    체크포인트 경로가 하이픈/언더스코어로 혼용될 수 있어 보정 시도.
    예: yolo11n-safety.pt 또는 yolo11n_safety.pt
    """
    p = _resolve(path_str)
    if p.exists():
        return p
    # 대체 케이스 시도
    alt = Path(str(p).replace("n-safety", "n_safety"))
    if alt.exists():
        return alt
    alt2 = Path(str(p).replace("n_safety", "n-safety"))
    if alt2.exists():
        return alt2
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {p}")

# ---------------- I/O 유틸 ----------------
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

def list_videos_in_dir(video_dir: Path) -> List[Path]:
    vids: List[Path] = []
    for ext in VIDEO_EXTS:
        vids.extend(video_dir.rglob(f"*{ext}"))
    return sorted(vids)

def safe_crop(bgr: np.ndarray, box_xyxy, pad: float = 0.05) -> np.ndarray:
    """박스 주변을 살짝 여유(pad 비율) 주고 안전하게 자르기."""
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = map(float, box_xyxy)
    dw, dh = (x2 - x1), (y2 - y1)
    x1 -= dw * pad; y1 -= dh * pad
    x2 += dw * pad; y2 += dh * pad
    xi1 = max(0, int(np.floor(x1))); yi1 = max(0, int(np.floor(y1)))
    xi2 = min(w, int(np.ceil(x2)));  yi2 = min(h, int(np.ceil(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return bgr
    return bgr[yi1:yi2, xi1:xi2].copy()

# ---------------- 핵심 처리 ----------------
def process_video(
    video_path: str | Path,
    out_videos_dir: Path,
    out_crops_root: Path,
    out_logs_dir: Path,
    combined_csv_path: Path,  # 공통 CSV 파일 경로
    pose_ckpt: Path,
    ppe_ckpt: Path,
    tracker_yaml: str,
    imgsz: int,
    det_conf: float,
    kpt_thr: float,
    draw_kpts: bool,
    device: str,
    iou_thr: float = 0.05,
    vid_stride: int = 1,
    ppe_interval: int = 10,  # PPE 탐지 간격 (프레임)
    show: bool = False,
) -> Tuple[Path, Dict]:
    """단일 비디오 처리하고, 주석 오버레이된 mp4 경로 반환."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"비디오를 열 수 없어요: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 출력 경로들 준비
    out_videos_dir.mkdir(parents=True, exist_ok=True)
    out_logs_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_crops_root / video_path.stem
    crops_dir.mkdir(parents=True, exist_ok=True)

    out_mp4   = out_videos_dir / f"{video_path.stem}_safety.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (W, H))

    # 로그 CSV 초기화
    write_header = not combined_csv_path.exists()
    csv_f = open(combined_csv_path, "a", newline="", encoding="utf-8")  # append 모드로 열기
    csv_w = csv.writer(csv_f)
    if write_header:
        csv_w.writerow(["video_name", "frame", "id", "x1", "y1", "x2", "y2", "score", "helmet", "vest", "part", 
                       "helmet_bbox", "helmet_score", "vest_bbox", "vest_score"])  # 추가된 컬럼

    # 모델 로드 (CPU 전용)
    pose_model = YOLO(str(pose_ckpt))
    pose_model.cpu()  # 명시적으로 CPU 모드로 설정
    pose_model.model.half = False  # FP16 비활성화
    pose_model.model.fuse()  # 모델 퓨징
    
    ppe_model = PPEDetector(str(ppe_ckpt), conf=det_conf, device='cpu', imgsz=imgsz)
    tracker = Tracker(
        model=pose_model,
        tracker_name=tracker_yaml,
        img_size=imgsz,
        det_conf=det_conf,
        device='cpu',
    )

    # 성능 측정 준비
    start_time = time.time()
    frame_times = []
    processing_metrics = {
        "tracking_time": 0,
        "ppe_detection_time": 0,
        "drawing_time": 0
    }
    
    # PPE 상태와 crop 비디오 추적을 위한 딕셔너리 초기화
    ppe_states = {}
    person_crops = {}  # ID별 VideoWriter를 저장할 딕셔너리
    
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        frame_start = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # 프레임 스킵
        if vid_stride > 1 and (frame_idx % vid_stride != 0):
            video_writer.write(frame)
            if show:
                cv2.imshow("safety", frame)
                if cv2.waitKey(1) == 27: break
            continue

        # 1) 사람 추적
        track_start = time.time()
        t_res = tracker.track_frame(frame)
        processing_metrics["tracking_time"] += time.time() - track_start
        
        person_xyxys = t_res["xyxys"]         # (N,4)
        ids         = t_res["ids"]            # (N,)  compact id
        scores      = t_res["scores"]         # (N,)
        kpt_xy      = t_res["kpt_xy"]         # (N,17,2) or None
        kpt_conf    = t_res["kpt_conf"]       # (N,17)   or None

        # 2) PPE 탐지 (ppe_interval 프레임마다 실행)
        ppe_start = time.time()
        if frame_idx % ppe_interval == 0:
            ppe_res = ppe_model.infer(frame)
            # 현재 프레임의 person boxes와 매칭
            current_ppe_list = assign_ppe_to_person_boxes(ppe_res, person_xyxys, iou_thr=iou_thr)
            # ID별 PPE 상태 업데이트
            for i, person_data in enumerate(current_ppe_list):
                if i < len(ids) and ids[i] is not None:
                    person_id = int(ids[i])
                    ppe_states[person_id] = {
                        'helmet': person_data['helmet'],
                        'vest': person_data['vest'],
                        'last_updated': frame_idx
                    }
        processing_metrics["ppe_detection_time"] += time.time() - ppe_start

        # 3) 각 person에 대해 가장 최근의 PPE 상태 적용
        ppe_list = []
        for i, _ in enumerate(person_xyxys):
            person_id = int(ids[i]) if (ids is not None and i < len(ids) and ids[i] is not None) else None
            if person_id is not None and person_id in ppe_states:
                ppe_list.append({
                    'helmet': ppe_states[person_id]['helmet'],
                    'vest': ppe_states[person_id]['vest']
                })
            else:
                ppe_list.append({'helmet': None, 'vest': None})

        # 4) 그리기 + 크롭 저장 + CSV 로그
        draw_start = time.time()
        for i, pbox in enumerate(person_xyxys):
            pid   = int(ids[i]) if ids is not None and i < len(ids) else None
            score = float(scores[i]) if i < len(scores) else 0.0
            kxy   = kpt_xy[i]   if kpt_xy   is not None and i < len(kpt_xy)   else None
            kcf   = kpt_conf[i] if kpt_conf is not None and i < len(kpt_conf) else None
            helmet= ppe_list[i]["helmet"] if i < len(ppe_list) else None
            vest  = ppe_list[i]["vest"]   if i < len(ppe_list) else None

            # draw
            draw_person(frame, pbox, track_id=pid,
                        kpt_xy=kxy, kpt_conf=kcf, kpt_thr=kpt_thr, draw_kpts=draw_kpts,
                        helmet=helmet, vest=vest)

            # crop 저장 (bbox_id 별 비디오)
            pid = int(ids[i]) if (ids is not None and i < len(ids) and ids[i] is not None) else None
            if pid is not None:
                if pid not in person_crops:
                    # 새로운 ID가 등장하면 해당 ID의 VideoWriter 생성
                    crop = safe_crop(frame, pbox, pad=0.05)
                    h, w = crop.shape[:2]
                    crops_dir.mkdir(parents=True, exist_ok=True)
                    out_path = str(crops_dir / f"id_{pid:03d}_track.mp4")
                    person_crops[pid] = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    
                if pid in person_crops:
                    # 기존 ID의 경우 프레임 추가
                    crop = safe_crop(frame, pbox, pad=0.05)
                    if crop.size > 0:  # 유효한 크기의 크롭인 경우만 저장
                        person_crops[pid].write(crop)

            # CSV 로그
            x1, y1, x2, y2 = map(int, pbox)
            part = get_person_part(kxy, kcf, kpt_thr) if (kxy is not None and kcf is not None) else "unknown"
            
            # PPE 검출 정보 추출
            helmet_bbox = "None"
            helmet_score = "None"
            vest_bbox = "None"
            vest_score = "None"
            
            if person_id is not None and person_id in ppe_states:
                if 'helmet' in ppe_states[person_id] and ppe_states[person_id]['helmet'] is not None:
                    helmet_bbox = json.dumps(ppe_states[person_id].get('helmet_bbox', []))
                    helmet_score = f"{ppe_states[person_id].get('helmet_score', 0.0):.4f}"
                if 'vest' in ppe_states[person_id] and ppe_states[person_id]['vest'] is not None:
                    vest_bbox = json.dumps(ppe_states[person_id].get('vest_bbox', []))
                    vest_score = f"{ppe_states[person_id].get('vest_score', 0.0):.4f}"
            
            csv_w.writerow([
                video_path.stem,  # 비디오 이름 추가
                frame_idx, pid if pid is not None else -1,
                x1, y1, x2, y2,
                f"{score:.4f}",
                ("True" if helmet is True else ("False" if helmet is False else "None")),
                ("True" if vest   is True else ("False" if vest   is False else "None")),
                part,
                helmet_bbox,
                helmet_score,
                vest_bbox,
                vest_score
            ])

        # 출력 프레임 기록
        video_writer.write(frame)
        if show:
            cv2.imshow("safety", frame)
            if cv2.waitKey(1) == 27: break

        processing_metrics["drawing_time"] += time.time() - draw_start
        frame_times.append(time.time() - frame_start)
        
        # 진행 상황 로깅 (10% 단위)
        if frame_idx % max(1, total_frames // 10) == 0:
            progress = (frame_idx / total_frames) * 100
            logging.info(f"Processing: {progress:.1f}% complete")

    # 성능 메트릭 계산
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_idx / total_time
    
    performance_metrics = {
        "total_frames": frame_idx,
        "total_time": total_time,
        "fps": fps,
        "avg_frame_time": np.mean(frame_times) if frame_times else 0,
        "tracking_time": processing_metrics["tracking_time"],
        "ppe_detection_time": processing_metrics["ppe_detection_time"],
        "drawing_time": processing_metrics["drawing_time"],
        "ppe_detection_intervals": ppe_interval,
        "ppe_detections_performed": frame_idx // ppe_interval
    }
    
    # 성능 로그 저장
    perf_log_path = out_logs_dir / f"{video_path.stem}_performance.csv"
    with open(perf_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in performance_metrics.items():
            writer.writerow([key, f"{value:.3f}"])
    
    # 성능 정보 로깅
    logging.info(f"\nVideo Processing Performance Metrics:")
    logging.info(f"Total Frames: {frame_idx}")
    logging.info(f"Total Processing Time: {total_time:.2f} seconds")
    logging.info(f"Average FPS: {fps:.2f}")
    logging.info(f"Average Frame Processing Time: {performance_metrics['avg_frame_time']*1000:.2f} ms")
    logging.info(f"Time Breakdown:")
    logging.info(f"  - Tracking: {processing_metrics['tracking_time']:.2f} seconds")
    logging.info(f"  - PPE Detection: {processing_metrics['ppe_detection_time']:.2f} seconds")
    logging.info(f"  - Drawing and Saving: {processing_metrics['drawing_time']:.2f} seconds")
    
    # 모든 비디오 작성기 해제
    cap.release()
    video_writer.release()
    
    # 모든 개별 추적 비디오 작성기 해제
    for writer in person_crops.values():
        writer.release()
    
    csv_f.close()
    if show:
        cv2.destroyAllWindows()
        
    # 생성된 추적 비디오 정보 추가
    performance_metrics["tracked_persons"] = len(person_crops)
    
    return out_mp4, performance_metrics

def process_folder():
    """VIDEO_DIR 아래 모든 비디오를 일괄 처리."""
    video_dir = _resolve(VIDEO_DIR)
    out_root  = _resolve(OUT_DIR)
    out_videos_dir = out_root / "videos"
    out_crops_root = out_root / "crops"
    out_logs_dir   = out_root / "logs"

    # 전체 결과를 저장할 CSV 파일 생성
    out_logs_dir.mkdir(parents=True, exist_ok=True)
    tracks_csv_path = out_logs_dir / "tracks.csv"
    perf_csv_path = out_logs_dir / "performance.csv"
    
    # tracks.csv가 없을 경우에만 헤더 작성
    if not tracks_csv_path.exists():
        with open(tracks_csv_path, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["video_name", "frame", "id", "x1", "y1", "x2", "y2", "score", "helmet", "vest", "part"])

    pose_ckpt = _resolve_ckpt(MODEL_PEOPLE_PATH)
    ppe_ckpt  = _resolve_ckpt(MODEL_PPE_PATH)

    videos = list_videos_in_dir(video_dir)
    if not videos:
        print(f"[WARN] 비디오가 없습니다: {video_dir}")
        return []

    saved_paths: List[Path] = []
    all_performance_metrics = []
    
    for vp in videos:
        print(f"[INFO] 처리 중: {vp}")
        out_mp4, perf_metrics = process_video(
            video_path=vp,
            out_videos_dir=out_videos_dir,
            out_crops_root=out_crops_root,
            out_logs_dir=out_logs_dir,
            combined_csv_path=tracks_csv_path,  # 수정된 부분
            pose_ckpt=pose_ckpt,
            ppe_ckpt=ppe_ckpt,
            tracker_yaml=TRACKER_YAML,
            imgsz=IMG_SIZE,
            det_conf=DET_CONF,
            kpt_thr=KPT_CONF,
            draw_kpts=DRAW_KPTS,
            device=DEVICE,
            iou_thr=0.05,
            vid_stride=1,
            show=False,
        )
        print(f"   -> Saved video: {out_mp4}")
        saved_paths.append(out_mp4)
        
        # 성능 메트릭에 비디오 이름 추가
        perf_metrics['video_name'] = vp.stem
        all_performance_metrics.append(perf_metrics)
        
        # 각 비디오의 성능 메트릭을 performance.csv에 추가
        write_header = not perf_csv_path.exists()
        with open(perf_csv_path, "a", newline="", encoding="utf-8") as f:
            fieldnames = ['video_name'] + list(perf_metrics.keys() - {'video_name'})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(perf_metrics)
            
    return saved_paths

if __name__ == "__main__":
    process_folder()
