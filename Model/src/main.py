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
    # 원본 프레임을 저장하는 writer
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # bbox_writer 제거: 원본 영상에 바운딩 박스를 그리는 방식으로 변경
    # mota 평가를 위한 코드 추가
    mot_dir = os.path.join(OUT_DIR, "tracking_mot")
    os.makedirs(mot_dir, exist_ok=True)
    mot_output_path = os.path.join(mot_dir, f"{base}.txt")
    mot_file = open(mot_output_path, "w")

    # 크롭된 바운딩 박스 영상들을 개별 파일로 저장
    bbox_dir = os.path.join(OUT_DIR, "bbox_crops")
    os.makedirs(bbox_dir, exist_ok=True)
    video_crop_dir = os.path.join(bbox_dir, base)
    os.makedirs(video_crop_dir, exist_ok=True)
    print(f"크롭된 바운딩 박스 영상들 저장 디렉토리: {video_crop_dir}")

    print(f"처리 중: {video_path}")
    print(f"총 프레임: {total}, FPS: {fps}")
    print(f"MOTA 결과 저장 경로: {mot_output_path}")
    
    last_helmet_by_id = {}

    # 각 ID별로 VideoWriter를 관리하는 딕셔너리
    id_writers = {}
    id_crop_sizes = {}  # 각 ID별 첫 번째 크롭 크기 저장

    frames_with_detection = 0
    frame_idx = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # 원본 프레임에 바운딩 박스를 그릴 것이므로 검은색 배경 프레임 생성 로직 제거
        
        #1) 사람 트래킹
        det = tracker.track_frame(frame)
        xyxys, scores, ids = det["xyxys"], det["scores"], det["ids"]
        kpt_xy, kpt_conf   = det["kpt_xy"], det["kpt_conf"]
        
        #MOTA 평가를 위한 코드 추가
        if ids is not None:
            for i, track_id in enumerate(ids):
                # xyxy -> xywh 변환
                x1, y1, x2, y2 = xyxys[i]
                w, h = x2 - x1, y2 - y1
                # MOT Challenge 포맷: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, -1, -1, -1, -1
                mot_file.write(f"{frame_idx},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},-1,-1,-1,-1\n")

        frame_has_detection = False
        N = xyxys.shape[0]
        
        #2) helmet Detection
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
                #여기서 절대 크래시하지 않음: carry 로 대체
                print(f"[Frame {frame_idx}] Helmet detection 실패: {type(e).__name__}: {e}")
                helmet_boxes = None  # 아래에서 carry 사용
        else:
            print(f"[Frame {frame_idx}] Helmet detection SKIP (carry 사용)")

        # 크롭된 이미지들을 저장할 리스트
        cropped_frames = []

        for i in range(N):
            this_kp_xy   = kpt_xy[i]   if kpt_xy   is not None and i < kpt_xy.shape[0]   else None
            this_kp_conf = kpt_conf[i] if kpt_conf is not None and i < kpt_conf.shape[0] else None
            this_id      = int(ids[i]) if ids is not None and i < len(ids)               else None

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

                # 바운딩 박스 영역 크롭 (패딩 추가) - 크롭 영상 저장용 로직
                x1, y1, x2, y2 = map(int, xyxys[i])
                
                # 패딩 계산 (바운딩 박스 크기의 15% 정도)
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                padding_x = int(bbox_w * 0.15)
                padding_y = int(bbox_h * 0.2)  # 위쪽에 텍스트 공간을 위해 조금 더 여유
                
                # 패딩을 적용한 확장된 바운딩 박스
                padded_x1 = max(0, x1 - padding_x)
                padded_y1 = max(0, y1 - padding_y)
                padded_x2 = min(width, x2 + padding_x)
                padded_y2 = min(height, y2 + padding_y)
                
                # 유효한 바운딩 박스인지 확인
                if padded_x2 > padded_x1 and padded_y2 > padded_y1:
                    # 원본 이미지에서 패딩이 적용된 바운딩 박스 영역 크롭
                    cropped = frame[padded_y1:padded_y2, padded_x1:padded_x2].copy()
                    crop_h, crop_w = cropped.shape[:2]
                    
                    # 헬멧 착용 여부에 따른 색상 설정
                    if has_helmet is True:
                        color = (0, 255, 0)  # 초록색 (헬멧 착용)
                        helmet_text = "Helmet"
                    elif has_helmet is False:
                        color = (0, 0, 255)  # 빨간색 (헬멧 미착용)
                        helmet_text = "No Helmet"
                    else:
                        color = (0, 255, 255)  # 노란색 (불확실)
                        helmet_text = "Unknown"
                    
                    # ID와 헬멧 정보를 크롭된 이미지에 추가
                    if this_id is not None:
                        # 텍스트 크기를 크롭된 이미지 크기에 맞게 조정
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        crop_h, crop_w = cropped.shape[:2]
                        font_scale = min(crop_w, crop_h) / 300.0
                        font_scale = max(0.4, min(font_scale, 1.2))
                        thickness = max(1, int(font_scale * 2))
                        
                        # ID 텍스트
                        id_text = f"ID: {this_id}"
                        (id_w, id_h), _ = cv2.getTextSize(id_text, font, font_scale, thickness)
                        
                        # 헬멧 텍스트
                        (helmet_w, helmet_h), _ = cv2.getTextSize(helmet_text, font, font_scale, thickness)
                        
                        # 텍스트 배경
                        max_width = max(id_w, helmet_w)
                        total_height = id_h + helmet_h + 10
                        
                        # 텍스트 배경 그리기
                        cv2.rectangle(cropped, 
                                      (5, 5), 
                                      (15 + max_width, 15 + total_height), 
                                      (40, 40, 40), -1)
                        
                        # ID 텍스트 그리기
                        cv2.putText(cropped, id_text, 
                                    (10, 15 + id_h), 
                                    font, font_scale, color, thickness)
                        
                        # 헬멧 상태 텍스트 그리기
                        cv2.putText(cropped, helmet_text, 
                                    (10, 25 + id_h + helmet_h), 
                                    font, font_scale, color, thickness)
                        
                        # 바운딩 박스 테두리 그리기 (선택사항)
                        cv2.rectangle(cropped, (0, 0), (crop_w-1, crop_h-1), color, 3)

                        # 원래 바운딩 박스 위치를 크롭된 이미지 내에서 표시
                        relative_x1 = x1 - padded_x1
                        relative_y1 = y1 - padded_y1
                        relative_x2 = x2 - padded_x1
                        relative_y2 = y2 - padded_y1
                        cv2.rectangle(cropped, (relative_x1, relative_y1), (relative_x2, relative_y2), color, 3)
                    
                    # 크롭된 이미지를 리스트에 추가 (동적 크기 유지)
                    cropped_frames.append(cropped)

                    # 각 ID별로 개별 영상 파일 생성/관리
                    if this_id is not None:
                        if this_id not in id_writers:
                            # 첫 번째 프레임의 크기로 VideoWriter 생성
                            crop_video_path = os.path.join(video_crop_dir, f"ID_{this_id}_crop.mp4")
                            id_writers[this_id] = cv2.VideoWriter(crop_video_path, fourcc, fps, (crop_w, crop_h))
                            id_crop_sizes[this_id] = (crop_w, crop_h)
                            print(f"ID {this_id} 크롭 영상 시작: {crop_video_path} (크기: {crop_w}x{crop_h})")
                        else:
                            # 기존 크기와 다르면 리사이즈 (비율 유지)
                            target_w, target_h = id_crop_sizes[this_id]
                            if crop_w != target_w or crop_h != target_h:
                                cropped = cv2.resize(cropped, (target_w, target_h))
                        
                        # 해당 ID의 영상에 프레임 추가
                        id_writers[this_id].write(cropped)

                # 기존 draw_person 함수 호출 
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
                    "helmet": helmet_json,  # "True" or "False" or "Skip"
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
    #MOTA 평가를 위한 코드 추가
    mot_file.close() 

    # ID별 개별 writer들을 닫는 코드 추가
    for id_writer in id_writers.values():
        id_writer.release()

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
