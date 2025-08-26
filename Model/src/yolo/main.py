from pathlib import Path
import argparse
import os,sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import *

from video import (
    process_video, list_videos_in_dir,  # 처리 함수/도우미
    _resolve, _resolve_ckpt             # 경로 보정 유틸( video.py 내부 제공 )
)

def run_single(video_path: Path,
               out_root: Path,
               pose_ckpt: Path,
               ppe_ckpt: Path,
               tracker_yaml: str,
               imgsz: int,
               det_conf: float,
               kpt_thr: float,
               draw_kpts: bool,
               device: str,
               iou_thr: float,
               vid_stride: int,
               show: bool):
    out_videos_dir = out_root / "videos"
    out_crops_root = out_root / "crops"
    out_logs_dir   = out_root / "logs"
    combined_csv_path = out_logs_dir / "tracks.csv"

    saved = process_video(
        video_path=video_path,
        out_videos_dir=out_videos_dir,
        out_crops_root=out_crops_root,
        out_logs_dir=out_logs_dir,
        combined_csv_path=combined_csv_path,
        pose_ckpt=pose_ckpt,
        ppe_ckpt=ppe_ckpt,
        tracker_yaml=tracker_yaml,
        imgsz=imgsz,
        det_conf=det_conf,
        kpt_thr=kpt_thr,
        draw_kpts=draw_kpts,
        device=device,
        iou_thr=iou_thr,
        vid_stride=vid_stride,
        show=show
    )
    print(f"✅ Saved video: {saved}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="단일 비디오 경로(지정 시 해당 파일만 처리). 미지정 시 VIDEO_DIR 전체 처리")
    ap.add_argument("--dir", help="비디오 폴더 경로(기본: config.VIDEO_DIR)")
    ap.add_argument("--out", help="출력 루트 폴더(기본: config.OUT_DIR)")
    ap.add_argument("--imgsz", type=int, help="이미지 사이즈(기본: config.IMG_SIZE)")
    ap.add_argument("--conf", type=float, help="탐지 confidence(기본: config.DET_CONF) — PPE도 동일값 사용")
    ap.add_argument("--kpt-thr", type=float, help="키포인트 표시 임계값(기본: config.KPT_CONF)")
    ap.add_argument("--no-kpts", action="store_true", help="키포인트 점 표시하지 않기")
    ap.add_argument("--device", help="cpu / cuda:0 (기본: config.DEVICE)")
    ap.add_argument("--tracker", help="bytetrack.yaml / botsort.yaml (기본: config.TRACKER_YAML)")
    ap.add_argument("--stride", type=int, help="프레임 스킵(기본=1)")
    ap.add_argument("--iou-thr", type=float, help="PPE-사람 매칭 IoU 임계값(기본=0.05)")
    ap.add_argument("--show", action="store_true", help="윈도우로 미리보기")
    args = ap.parse_args()

    # --- 경로/설정 병합 (config 기본값 + CLI override) ---
    video_dir = _resolve(args.dir) if args.dir else _resolve(VIDEO_DIR)
    out_root  = _resolve(args.out) if args.out else _resolve(OUT_DIR)
    pose_ckpt = _resolve_ckpt(MODEL_PEOPLE_PATH)
    ppe_ckpt  = _resolve_ckpt(MODEL_PPE_PATH)

    imgsz     = args.imgsz   if args.imgsz   is not None else int(IMG_SIZE)
    det_conf  = args.conf    if args.conf    is not None else float(DET_CONF)
    kpt_thr   = args.kpt_thr if args.kpt_thr is not None else float(KPT_CONF)
    draw_kpts = False if args.no_kpts else bool(DRAW_KPTS)
    device    = args.device  if args.device  is not None else str(DEVICE)
    tracker_y = args.tracker if args.tracker is not None else TRACKER_YAML
    stride    = args.stride  if args.stride  is not None else 1
    iou_thr   = args.iou_thr if args.iou_thr is not None else 0.05
    show_flag = bool(args.show)

    # --- 실행 ---
    if args.video:
        vp = Path(args.video)
        print(f"[INFO] 단일 파일 처리: {vp}")
        run_single(
            video_path=vp,
            out_root=out_root,
            pose_ckpt=pose_ckpt,
            ppe_ckpt=ppe_ckpt,
            tracker_yaml=tracker_y,
            imgsz=imgsz,
            det_conf=det_conf,
            kpt_thr=kpt_thr,
            draw_kpts=draw_kpts,
            device=device,
            iou_thr=iou_thr,
            vid_stride=stride,
            show=show_flag
        )
    else:
        vids = list_videos_in_dir(video_dir)
        print(f"[INFO] 폴더 처리: {video_dir} (총 {len(vids)}개)")
        for vp in vids:
            print(f" - 처리 중: {vp}")
            run_single(
                video_path=vp,
                out_root=out_root,
                pose_ckpt=pose_ckpt,
                ppe_ckpt=ppe_ckpt,
                tracker_yaml=tracker_y,
                imgsz=imgsz,
                det_conf=det_conf,
                kpt_thr=kpt_thr,
                draw_kpts=draw_kpts,
                device=device,
                iou_thr=iou_thr,
                vid_stride=stride,
                show=show_flag
            )

if __name__ == "__main__":
    main()
