from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Callable
import csv
import cv2
import numpy as np
from ultralytics import YOLO
import os, sys
import time
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# project module
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import *                     
from tracker import *                    
from ppe_detection import *               
from people_detection import *       
from Model.utils.face_blur import estimate_face_area, apply_face_blur

# checkpoint 경로
def _resolve(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (Path(__file__).resolve().parent / p).resolve()

def _resolve_ckpt(path_str: str) -> Path:
    p = _resolve(path_str)
    if p.exists():
        return p
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

# crop
def safe_crop(bgr: np.ndarray, box_xyxy, pad: float = 0.05, return_rect: bool = False, top_pad_pixels: int = 0):
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = map(float, box_xyxy)
    dw, dh = (x2 - x1), (y2 - y1)
    x1 -= dw * pad; y1 -= dh * pad
    x2 += dw * pad; y2 += dh * pad

    y1 -= (dh * pad + top_pad_pixels)

    xi1 = max(0, int(np.floor(x1))); yi1 = max(0, int(np.floor(y1)))
    xi2 = min(w, int(np.ceil(x2)));  yi2 = min(h, int(np.ceil(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        crop = bgr
        rect = (0, 0, w, h)
    else:
        crop = bgr[yi1:yi2, xi1:xi2].copy()
        rect = (xi1, yi1, xi2, yi2)
    return (crop, rect) if return_rect else crop

# ---- 크기 고정 및 VideoWriter 보조 ----
def _even_size(w: int, h: int) -> Tuple[int, int]:
    return (int(np.ceil(w / 2.0) * 2), int(np.ceil(h / 2.0) * 2))

def _fit_to_size_letterbox(img: np.ndarray, out_size: Tuple[int, int], pad_color=(0,0,0)) -> np.ndarray:
    out_w, out_h = out_size
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    scale = min(out_w / w, out_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((out_h, out_w, 3), pad_color, dtype=img.dtype)
    x = (out_w - nw) // 2
    y = (out_h - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas

def _open_writer_with_fallbacks(path: str, fps: float, size: Tuple[int,int]) -> cv2.VideoWriter:
    for fourcc_str in ("mp4v", "avc1", "H264", "XVID"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_str), fps, size)
        if vw.isOpened():
            return vw
    raise RuntimeError(f"VideoWriter를 열 수 없습니다(size={size}, fps={fps})")

def process_video(
    video_path: str | Path,
    out_videos_dir: Path,
    out_crops_root: Path,
    out_logs_dir: Path,
    combined_csv_path: Path,
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
    ppe_interval: int = 10,
    show: bool = False,
    person_history: Dict[int, "Person"] | None = None,
    get_person_part_with_history: Optional[Callable] = None,
) -> Tuple[Path, Dict]:

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"비디오를 열 수 없어요: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # output_dir
    out_videos_dir.mkdir(parents=True, exist_ok=True)
    out_logs_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_crops_root / video_path.stem
    crops_dir.mkdir(parents=True, exist_ok=True)

    out_mp4   = out_videos_dir / f"{video_path.stem}_output.mp4"
    video_writer = _open_writer_with_fallbacks(str(out_mp4), fps, (W, H))

    # 로그 CSV 
    write_header = not combined_csv_path.exists()
    csv_f = open(combined_csv_path, "a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    if write_header:
        csv_w.writerow([
            "video_name", "frame", "id", "x1", "y1", "x2", "y2",
            "helmet", "vest", "part",
            "helmet_bbox",  "vest_bbox"
        ])

    # 모델 로드 
    pose_model = YOLO(str(pose_ckpt), verbose=False)
    pose_model.cpu()
    pose_model.model.half = False
    pose_model.model.fuse()

    ppe_model = PPEDetector(str(ppe_ckpt), conf=det_conf, device='cpu', imgsz=imgsz)
    tracker = Tracker(model=pose_model, tracker_name=tracker_yaml,
                      img_size=imgsz, det_conf=det_conf, device='cpu')

    # 성능 측정 초기화 
    start_ts = time.perf_counter()                          
    frame_time_accum_s = 0.0                                 
    processing_metrics = {"tracking_time": 0.0, "ppe_detection_time": 0.0, "drawing_time": 0.0}

    people_calls = 0                                        
    ppe_calls = 0   
    
    # 상태
    ppe_states: Dict[int, dict] = {}
    person_crops: Dict[int, cv2.VideoWriter] = {}
    person_sizes: Dict[int, Tuple[int,int]] = {}   

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        frame_start = time.perf_counter()
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
            frame_time_accum_s += (time.perf_counter() - frame_start)
            continue

        # 1) 사람 추적
        t0 = time.perf_counter()
        t_res = tracker.track_frame(frame)
        processing_metrics["tracking_time"] += time.perf_counter() - t0

        person_xyxys = t_res["xyxys"]   
        ids          = t_res["ids"]    
        kpt_xy       = t_res["kpt_xy"]  
        kpt_conf     = t_res["kpt_conf"]
        
        # Detection이 없어도 기록
        N = 0 if person_xyxys is None else int(person_xyxys.shape[0])
        if N == 0:
            video_writer.write(frame)
            frame_time_accum_s += (time.perf_counter() - frame_start)
            continue

        # 2) PPE 탐지 
        t0 = time.perf_counter()
        if frame_idx % ppe_interval == 0:
            ppe_res = ppe_model.infer(frame)
            matched = assign_ppe_to_person_boxes(ppe_res, person_xyxys, iou_thr=iou_thr)
            for i, person_data in enumerate(matched):
                if i < len(ids) and ids[i] is not None:
                    pid = int(ids[i])
                    ppe_states[pid] = {
                        'helmet': person_data.get('helmet'),
                        'vest': person_data.get('vest'),
                        'helmet_bbox': person_data.get('helmet_bbox'),
                        'vest_bbox': person_data.get('vest_bbox'),
                        'last_updated': frame_idx
                    }
            ppe_calls += 1
        processing_metrics["ppe_detection_time"] += time.perf_counter() - t0

        # 3) 최신 PPE 상태 적용
        ppe_list = []
        for i in range(N):
            pid = int(ids[i]) if (ids is not None and ids[i] is not None) else None
            if pid is not None and pid in ppe_states:
                ppe_list.append({'helmet': ppe_states[pid]['helmet'],
                                 'vest': ppe_states[pid]['vest']})
            else:
                ppe_list.append({'helmet': None, 'vest': None})

        # 4) BBox + CSV
        t0 = time.perf_counter()
        for i in range(N):
            pbox  = person_xyxys[i]
            pid   = int(ids[i]) if (ids is not None and ids[i] is not None) else None
            kxy   = kpt_xy[i]   if (kpt_xy   is not None and i < len(kpt_xy))   else None
            kcf   = kpt_conf[i] if (kpt_conf is not None and i < len(kpt_conf)) else None
            helmet= ppe_list[i]["helmet"]
            vest  = ppe_list[i]["vest"]

            # 메인 프레임 오버레이(메인은 블러 X)
            draw_person(frame, pbox, track_id=pid,
                        kpt_xy=kxy, kpt_conf=kcf, kpt_thr=kpt_thr, draw_kpts=draw_kpts,
                        helmet=helmet, vest=vest)

            # 파트 감지 알림
            if pid is not None and person_history is not None and get_person_part_with_history is not None:
                if pid not in person_history:
                    person_history[pid] = Person(pid)
                current_part, count = get_person_part_with_history(kxy, kcf, kpt_thr, person_history[pid])
                if count >= 10:
                    logging.info(f"[알림] ID {pid}의 {current_part}가 {count}프레임 이상 연속 감지.")

            # 사람 Crop 영상 Blur 처리
            if pid is not None:
                top_margin_for_text = 50  
                crop, crop_rect = safe_crop(frame, pbox, pad=0.05, return_rect=True, top_pad_pixels=top_margin_for_text)

                # 얼굴 블러
                if kxy is not None:
                    if kcf is not None and kcf.ndim == 1:
                        kp = np.concatenate([kxy.astype(float), kcf.reshape(-1,1).astype(float)], axis=1)
                    elif kcf is not None and kcf.ndim == 2:
                        kp = np.concatenate([kxy.astype(float), kcf.astype(float)[...,None]], axis=1)
                    else:
                        kp = kxy.astype(float)
                    face_area_global = estimate_face_area(kp, pbox)
                    if face_area_global is not None:
                        fx1, fy1, fx2, fy2 = map(int, face_area_global)
                        cx1, cy1, cx2, cy2 = crop_rect
                        rx1 = max(0, fx1 - cx1); ry1 = max(0, fy1 - cy1)
                        rx2 = min(crop.shape[1], fx2 - cx1); ry2 = min(crop.shape[0], fy2 - cy1)
                        if rx2 > rx1 and ry2 > ry1:
                            crop = apply_face_blur(crop, (rx1, ry1, rx2, ry2))

                # crop 영상 기록 
                if pid not in person_crops:
                    h0, w0 = crop.shape[:2]
                    out_w, out_h = _even_size(w0, h0)
                    person_sizes[pid] = (out_w, out_h)
                    out_path = str(crops_dir / f"id_{pid:03d}_track.mp4")
                    person_crops[pid] = _open_writer_with_fallbacks(out_path, fps, (out_w, out_h))

                target_size = person_sizes[pid]
                crop_to_write = _fit_to_size_letterbox(crop, target_size, pad_color=(0,0,0))
                if person_crops[pid].isOpened():
                    person_crops[pid].write(crop_to_write)
                else:
                    logging.warning(f"[CROP] writer not opened for id={pid}")

            # csv 로그
            x1, y1, x2, y2 = map(int, pbox)
            part = get_person_part(kxy, kcf, kpt_thr) if (kxy is not None and kcf is not None) else "unknown"
            
            helmet_bbox = "None"
            vest_bbox   = "None"
            if pid is not None and pid in ppe_states:
                st = ppe_states[pid]
                if st.get('helmet_bbox') is not None:
                    helmet_bbox = json.dumps(st['helmet_bbox'])
                if st.get('vest_bbox') is not None:
                    vest_bbox = json.dumps(st['vest_bbox'])

            csv_w.writerow([
                video_path.stem, frame_idx, pid if pid is not None else -1,
                x1, y1, x2, y2,
                ("True" if helmet is True else ("False" if helmet is False else "None")),
                ("True" if vest   is True else ("False" if vest   is False else "None")),
                part, helmet_bbox, vest_bbox
            ])

        # 메인 프레임 기록
        video_writer.write(frame)
        if show:
            cv2.imshow("safety", frame)
            if cv2.waitKey(1) == 27: break

        processing_metrics["drawing_time"] += time.perf_counter() - t0


    # 성능 메트릭 계산
    elapsed_s = max(1e-9, time.perf_counter() - start_ts)                 # CHANGED
    fps_eff   = frame_idx   / elapsed_s                                   # CHANGED: 전체 프레임 소비 속도
    people_fps= people_calls / elapsed_s                                   # CHANGED: 사람 단계 호출 빈도
    ppe_fps   = ppe_calls    / elapsed_s                                   # CHANGED: PPE 단계 호출 빈도

    avg_frame_ms = (frame_time_accum_s / frame_idx * 1000.0) if frame_idx else 0.0  # CHANGED: 평균 프레임 처리시간(ms)

    
    performance_metrics = {
        "total_frames": frame_idx,
        "fps": fps_eff,
        "ppe_fps": ppe_fps,
        "people_fps": people_fps,
    }

    # 성능 로그 저장
    perf_log_path = out_logs_dir / f"{video_path.stem}_performance.csv"
    with open(perf_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["Metric", "Value"])
        for k, v in performance_metrics.items():
            writer.writerow([k, f"{v:.3f}"])

    # 해제
    cap.release()
    video_writer.release()
    for w in person_crops.values():
        w.release()
    csv_f.close()
    if show:
        cv2.destroyAllWindows()

    performance_metrics["tracked_persons"] = len(person_crops)
    return out_mp4, performance_metrics

# 전체 폴더 처리
def process_folder():
    video_dir = _resolve(VIDEO_DIR)
    out_root  = _resolve(OUT_DIR)
    out_videos_dir = out_root / "videos"
    out_crops_root = out_root / "crops"
    out_logs_dir   = out_root / "logs"

    out_logs_dir.mkdir(parents=True, exist_ok=True)
    tracks_csv_path = out_logs_dir / "tracks.csv"
    perf_csv_path   = out_logs_dir / "performance.csv"

    # tracks.csv 헤더
    if not tracks_csv_path.exists():
        with open(tracks_csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["video_name","frame","id","x1","y1","x2","y2","helmet","vest","part",
                 "helmet_bbox","vest_bbox"]
            )

    pose_ckpt = _resolve_ckpt(MODEL_PEOPLE_PATH)
    ppe_ckpt  = _resolve_ckpt(MODEL_PPE_PATH)

    videos = list_videos_in_dir(video_dir)

    saved_paths: List[Path] = []
    for vp in videos:
        print(f"[INFO] 처리 중: {vp}")
        out_mp4, perf_metrics = process_video(
            video_path=vp,
            out_videos_dir=out_videos_dir,
            out_crops_root=out_crops_root,
            out_logs_dir=out_logs_dir,
            combined_csv_path=tracks_csv_path,
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

        # performance.csv 누적
        write_header = not perf_csv_path.exists()
        with open(perf_csv_path, "a", newline="", encoding="utf-8") as f:
            fieldnames = ['video_name'] + [k for k in perf_metrics.keys()]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            row = {"video_name": vp.stem}
            row.update(perf_metrics)
            writer.writerow(row)

    return saved_paths

if __name__ == "__main__":
    process_folder()
