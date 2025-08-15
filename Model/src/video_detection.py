# pose_filter_with_bytetrack.py
# pip install ultralytics opencv-python

import os, json, glob
import cv2
from ultralytics import YOLO

# --------------------- 설정 ---------------------
MODEL_PATH = "Model/src/checkpoint/yolo11n-pose.pt"  # 또는 "yolo11n-pose.pt"  -> 내 경로로 바꾸기
VIDEO_DIR  = "../../Sample Data/Sample/01.원천데이터/video"            # 동영상 폴더
OUT_DIR    = "../../Sample Data/Sample/01.원천데이터/outputs"                 # 결과 저장 폴더
IMG_SIZE   = 640                                  # 추론 해상도
DET_CONF   = 0.25                                 # 박스 confidence 임계값
KPT_CONF   = 0.3                                  # 키포인트 confidence 임계값
DRAW_KPTS  = True                                 # keypoint도 함께 그림
# ------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "video"), exist_ok=True)
jsonl_path = os.path.join(OUT_DIR, "detections.jsonl")

# COCO 17 기준 인덱스 (Ultralytics pose 기본)
# 얼굴: 0: nose, 1: left_eye, 2: right_eye
# 어깨: 5: left_shoulder, 6: right_shoulder
# 발목: 15: left_ankle, 16: right_ankle
NOSE_IDX           = 0
LEFT_EYE_IDX       = 1
RIGHT_EYE_IDX      = 2
LEFT_SHOULDER_IDX  = 5
RIGHT_SHOULDER_IDX = 6
LEFT_ANKLE_IDX     = 15
RIGHT_ANKLE_IDX    = 16

def person_visible(kpt_xy, kpt_conf, thr=KPT_CONF):
    """
    kpt_xy: (K, 2), kpt_conf: (K,) or None
    사람의 어떤 부위든 충분히 보이는지 확인
    - 얼굴: 코/눈 중 1개 이상 (클로즈업)
    - 상반신: 어깨 중 1개 이상 (상반신 줌)
    - 하반신: 발목 중 1개 이상 (하반신 줌)
    """
    if kpt_xy is None or kpt_conf is None:
        return False
    K = kpt_xy.shape[0]
    if K <= max(NOSE_IDX, LEFT_EYE_IDX, RIGHT_EYE_IDX, LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX, LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX):
        return False
    
    # 얼굴 체크 (클로즈업 대응)
    nose_ok = kpt_conf[NOSE_IDX] is not None and kpt_conf[NOSE_IDX] >= thr
    left_eye_ok = kpt_conf[LEFT_EYE_IDX] is not None and kpt_conf[LEFT_EYE_IDX] >= thr
    right_eye_ok = kpt_conf[RIGHT_EYE_IDX] is not None and kpt_conf[RIGHT_EYE_IDX] >= thr
    face_ok = nose_ok or left_eye_ok or right_eye_ok
    
    # 어깨 체크 (상반신)
    ls_ok = kpt_conf[LEFT_SHOULDER_IDX] is not None and kpt_conf[LEFT_SHOULDER_IDX] >= thr
    rs_ok = kpt_conf[RIGHT_SHOULDER_IDX] is not None and kpt_conf[RIGHT_SHOULDER_IDX] >= thr
    shoulder_ok = ls_ok or rs_ok
    
    # 발목 체크 (하반신)
    la_ok = kpt_conf[LEFT_ANKLE_IDX] is not None and kpt_conf[LEFT_ANKLE_IDX] >= thr
    ra_ok = kpt_conf[RIGHT_ANKLE_IDX] is not None and kpt_conf[RIGHT_ANKLE_IDX] >= thr
    ankle_ok = la_ok or ra_ok
    
    # 얼굴 OR 어깨 OR 발목 중 하나라도 있으면 OK
    return face_ok or shoulder_ok or ankle_ok

def draw_person(frame, xyxy, track_id=None, kpt_xy=None, kpt_conf=None):
    x1, y1, x2, y2 = map(int, xyxy)
    
    # 트래킹 ID별로 색상 다르게 설정
    if track_id is not None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        color = colors[track_id % len(colors)]
    else:
        color = (0, 255, 0)
    
    # 바운딩박스 그리기
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # 트래킹 ID 표시
    if track_id is not None:
        cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if DRAW_KPTS and kpt_xy is not None and kpt_conf is not None:
        # 일반 키포인트들 (흰색 작은 점)
        for i, (x, y) in enumerate(kpt_xy):
            if kpt_conf[i] is None or kpt_conf[i] < KPT_CONF:
                continue
            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)
        
        # 얼굴 키포인트들 강조 (초록색 큰 점)
        face_indices = [NOSE_IDX, LEFT_EYE_IDX, RIGHT_EYE_IDX]
        if kpt_xy.shape[0] > max(face_indices):
            for idx in face_indices:
                if kpt_conf[idx] is not None and kpt_conf[idx] >= KPT_CONF:
                    cv2.circle(frame, (int(kpt_xy[idx,0]), int(kpt_xy[idx,1])), 5, (0, 255, 0), -1)
        
        # 어깨 키포인트들 강조 (파란색 큰 점)
        shoulder_indices = [LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX]
        if kpt_xy.shape[0] > max(shoulder_indices):
            for idx in shoulder_indices:
                if kpt_conf[idx] is not None and kpt_conf[idx] >= KPT_CONF:
                    cv2.circle(frame, (int(kpt_xy[idx,0]), int(kpt_xy[idx,1])), 5, (255, 0, 0), -1)
        
        # 발목 강조 (빨간색 큰 점)  
        if kpt_xy.shape[0] > max(LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX):
            for idx in (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX):
                if kpt_conf[idx] is not None and kpt_conf[idx] >= KPT_CONF:
                    cv2.circle(frame, (int(kpt_xy[idx,0]), int(kpt_xy[idx,1])), 5, (0, 0, 255), -1)

def process_video(video_path, model, jsonl_file):
    """동영상을 프레임별로 처리하고 결과 동영상 저장 (ByteTrack 사용)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"동영상을 열 수 없습니다: {video_path}")
        return
    
    # 동영상 정보 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 출력 동영상 설정
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = os.path.join(OUT_DIR, "video", f"{video_name}_tracked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    frames_with_detection = 0
    
    print(f"처리 중: {video_path}")
    print(f"총 프레임: {total_frames}, FPS: {fps}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # YOLO로 pose detection + tracking 실행 (ByteTrack 내장)
        results = model.track(frame, imgsz=IMG_SIZE, conf=DET_CONF, verbose=False, 
                             tracker="bytetrack.yaml", persist=True)
        result = results[0]
        
        boxes = result.boxes
        kpts = result.keypoints
        
        frame_has_detection = False
        
        if boxes is not None and len(boxes) > 0:
            xyxys = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            # 트래킹 ID 가져오기
            track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None
            
            # pose keypoints
            kp_xy = kpts.xy.cpu().numpy() if kpts is not None and kpts.xy is not None else None
            kp_conf = kpts.conf.cpu().numpy() if kpts is not None and kpts.conf is not None else None
            
            for i, xyxy in enumerate(xyxys):
                this_kp_xy = kp_xy[i] if kp_xy is not None and i < kp_xy.shape[0] else None
                this_kp_conf = kp_conf[i] if kp_conf is not None and i < kp_conf.shape[0] else None
                this_track_id = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else None
                
                # 사람의 어떤 부위든 보이면 처리 (얼굴/어깨/발목)
                if person_visible(this_kp_xy, this_kp_conf, KPT_CONF):
                    frame_has_detection = True
                    draw_person(frame, xyxy, this_track_id, this_kp_xy, this_kp_conf)
                    
                    # JSONL에 저장 (트래킹 ID 포함)
                    row = {
                        "video_path": os.path.abspath(video_path),
                        "frame_number": frame_count,
                        "track_id": this_track_id,  # 트래킹 ID 추가
                        "bbox_xyxy": [float(x) for x in xyxy.tolist()],
                        "score": float(scores[i]),
                        "kpts_xy": None if this_kp_xy is None else [[float(a), float(b)] for a,b in this_kp_xy.tolist()],
                        "kpts_conf": None if this_kp_conf is None else [None if c is None else float(c) for c in this_kp_conf.tolist()],
                    }
                    jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        if frame_has_detection:
            frames_with_detection += 1
            
        # 모든 프레임을 출력 동영상에 저장
        out_video.write(frame)
        
        # 진행률 출력
        if frame_count % 30 == 0:  # 30프레임마다 출력
            progress = (frame_count / total_frames) * 100
            print(f"진행률: {progress:.1f}% ({frame_count}/{total_frames})")
    
    cap.release()
    out_video.release()
    
    print(f"완료: {video_path}")
    print(f"detection이 있는 프레임: {frames_with_detection}/{frame_count}")
    print(f"결과 동영상: {out_video_path}")
    print("-" * 50)

def main():
    model = YOLO(MODEL_PATH)
    
    # 동영상 파일 찾기
    video_paths = []
    for ext in ("*.mp4", "*.mov", "*.avi", "*.wmv", "*.mkv"):
        video_paths.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
    video_paths.sort()
    
    if not video_paths:
        print(f"동영상 파일을 찾을 수 없습니다: {VIDEO_DIR}")
        return
    
    print(f"찾은 동영상 파일: {len(video_paths)}개")
    
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for video_path in video_paths:
            process_video(video_path, model, jsonl_file)
    
    print(f"모든 처리 완료!")
    print(f"JSONL 파일: {jsonl_path}")

if __name__ == "__main__":
    main()# pose_filter_with_bytetrack.py
# pip install ultralytics opencv-python

import os, json, glob
import cv2
from ultralytics import YOLO

# --------------------- 설정 ---------------------
MODEL_PATH = "Model/src/checkpoint/yolo11n-pose.pt"  # 또는 "yolo11n-pose.pt"  -> 내 경로로 바꾸기
VIDEO_DIR  = "../../Sample Data/Sample/01.원천데이터/video"            # 동영상 폴더
OUT_DIR    = "../../Sample Data/Sample/01.원천데이터/outputs"                 # 결과 저장 폴더
IMG_SIZE   = 640                                  # 추론 해상도
DET_CONF   = 0.25                                 # 박스 confidence 임계값
KPT_CONF   = 0.3                                  # 키포인트 confidence 임계값
DRAW_KPTS  = True                                 # keypoint도 함께 그림
# ------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "video"), exist_ok=True)
jsonl_path = os.path.join(OUT_DIR, "detections.jsonl")

# COCO 17 기준 인덱스 (Ultralytics pose 기본)
# 얼굴: 0: nose, 1: left_eye, 2: right_eye
# 어깨: 5: left_shoulder, 6: right_shoulder
# 발목: 15: left_ankle, 16: right_ankle
NOSE_IDX           = 0
LEFT_EYE_IDX       = 1
RIGHT_EYE_IDX      = 2
LEFT_SHOULDER_IDX  = 5
RIGHT_SHOULDER_IDX = 6
LEFT_ANKLE_IDX     = 15
RIGHT_ANKLE_IDX    = 16

def person_visible(kpt_xy, kpt_conf, thr=KPT_CONF):
    """
    kpt_xy: (K, 2), kpt_conf: (K,) or None
    사람의 어떤 부위든 충분히 보이는지 확인
    - 얼굴: 코/눈 중 1개 이상 (클로즈업)
    - 상반신: 어깨 중 1개 이상 (상반신 줌)
    - 하반신: 발목 중 1개 이상 (하반신 줌)
    """
    if kpt_xy is None or kpt_conf is None:
        return False
    K = kpt_xy.shape[0]
    if K <= max(NOSE_IDX, LEFT_EYE_IDX, RIGHT_EYE_IDX, LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX, LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX):
        return False
    
    # 얼굴 체크 (클로즈업 대응)
    nose_ok = kpt_conf[NOSE_IDX] is not None and kpt_conf[NOSE_IDX] >= thr
    left_eye_ok = kpt_conf[LEFT_EYE_IDX] is not None and kpt_conf[LEFT_EYE_IDX] >= thr
    right_eye_ok = kpt_conf[RIGHT_EYE_IDX] is not None and kpt_conf[RIGHT_EYE_IDX] >= thr
    face_ok = nose_ok or left_eye_ok or right_eye_ok
    
    # 어깨 체크 (상반신)
    ls_ok = kpt_conf[LEFT_SHOULDER_IDX] is not None and kpt_conf[LEFT_SHOULDER_IDX] >= thr
    rs_ok = kpt_conf[RIGHT_SHOULDER_IDX] is not None and kpt_conf[RIGHT_SHOULDER_IDX] >= thr
    shoulder_ok = ls_ok or rs_ok
    
    # 발목 체크 (하반신)
    la_ok = kpt_conf[LEFT_ANKLE_IDX] is not None and kpt_conf[LEFT_ANKLE_IDX] >= thr
    ra_ok = kpt_conf[RIGHT_ANKLE_IDX] is not None and kpt_conf[RIGHT_ANKLE_IDX] >= thr
    ankle_ok = la_ok or ra_ok
    
    # 얼굴 OR 어깨 OR 발목 중 하나라도 있으면 OK
    return face_ok or shoulder_ok or ankle_ok

def draw_person(frame, xyxy, track_id=None, kpt_xy=None, kpt_conf=None):
    x1, y1, x2, y2 = map(int, xyxy)
    
    # 트래킹 ID별로 색상 다르게 설정
    if track_id is not None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        color = colors[track_id % len(colors)]
    else:
        color = (0, 255, 0)
    
    # 바운딩박스 그리기
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # 트래킹 ID 표시
    if track_id is not None:
        cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if DRAW_KPTS and kpt_xy is not None and kpt_conf is not None:
        # 일반 키포인트들 (흰색 작은 점)
        for i, (x, y) in enumerate(kpt_xy):
            if kpt_conf[i] is None or kpt_conf[i] < KPT_CONF:
                continue
            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)
        
        # 얼굴 키포인트들 강조 (초록색 큰 점)
        face_indices = [NOSE_IDX, LEFT_EYE_IDX, RIGHT_EYE_IDX]
        if kpt_xy.shape[0] > max(face_indices):
            for idx in face_indices:
                if kpt_conf[idx] is not None and kpt_conf[idx] >= KPT_CONF:
                    cv2.circle(frame, (int(kpt_xy[idx,0]), int(kpt_xy[idx,1])), 5, (0, 255, 0), -1)
        
        # 어깨 키포인트들 강조 (파란색 큰 점)
        shoulder_indices = [LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX]
        if kpt_xy.shape[0] > max(shoulder_indices):
            for idx in shoulder_indices:
                if kpt_conf[idx] is not None and kpt_conf[idx] >= KPT_CONF:
                    cv2.circle(frame, (int(kpt_xy[idx,0]), int(kpt_xy[idx,1])), 5, (255, 0, 0), -1)
        
        # 발목 강조 (빨간색 큰 점)  
        if kpt_xy.shape[0] > max(LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX):
            for idx in (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX):
                if kpt_conf[idx] is not None and kpt_conf[idx] >= KPT_CONF:
                    cv2.circle(frame, (int(kpt_xy[idx,0]), int(kpt_xy[idx,1])), 5, (0, 0, 255), -1)

def process_video(video_path, model, jsonl_file):
    """동영상을 프레임별로 처리하고 결과 동영상 저장 (ByteTrack 사용)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"동영상을 열 수 없습니다: {video_path}")
        return
    
    # 동영상 정보 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 출력 동영상 설정
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = os.path.join(OUT_DIR, "video", f"{video_name}_tracked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    frames_with_detection = 0
    
    print(f"처리 중: {video_path}")
    print(f"총 프레임: {total_frames}, FPS: {fps}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # YOLO로 pose detection + tracking 실행 (ByteTrack 내장)
        results = model.track(frame, imgsz=IMG_SIZE, conf=DET_CONF, verbose=False, 
                             tracker="bytetrack.yaml", persist=True)
        result = results[0]
        
        boxes = result.boxes
        kpts = result.keypoints
        
        frame_has_detection = False
        
        if boxes is not None and len(boxes) > 0:
            xyxys = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            # 트래킹 ID 가져오기
            track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None
            
            # pose keypoints
            kp_xy = kpts.xy.cpu().numpy() if kpts is not None and kpts.xy is not None else None
            kp_conf = kpts.conf.cpu().numpy() if kpts is not None and kpts.conf is not None else None
            
            for i, xyxy in enumerate(xyxys):
                this_kp_xy = kp_xy[i] if kp_xy is not None and i < kp_xy.shape[0] else None
                this_kp_conf = kp_conf[i] if kp_conf is not None and i < kp_conf.shape[0] else None
                this_track_id = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else None
                
                # 사람의 어떤 부위든 보이면 처리 (얼굴/어깨/발목)
                if person_visible(this_kp_xy, this_kp_conf, KPT_CONF):
                    frame_has_detection = True
                    draw_person(frame, xyxy, this_track_id, this_kp_xy, this_kp_conf)
                    
                    # JSONL에 저장 (트래킹 ID 포함)
                    row = {
                        "video_path": os.path.abspath(video_path),
                        "frame_number": frame_count,
                        "track_id": this_track_id,  # 트래킹 ID 추가
                        "bbox_xyxy": [float(x) for x in xyxy.tolist()],
                        "score": float(scores[i]),
                        "kpts_xy": None if this_kp_xy is None else [[float(a), float(b)] for a,b in this_kp_xy.tolist()],
                        "kpts_conf": None if this_kp_conf is None else [None if c is None else float(c) for c in this_kp_conf.tolist()],
                    }
                    jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        if frame_has_detection:
            frames_with_detection += 1
            
        # 모든 프레임을 출력 동영상에 저장
        out_video.write(frame)
        
        # 진행률 출력
        if frame_count % 30 == 0:  # 30프레임마다 출력
            progress = (frame_count / total_frames) * 100
            print(f"진행률: {progress:.1f}% ({frame_count}/{total_frames})")
    
    cap.release()
    out_video.release()
    
    print(f"완료: {video_path}")
    print(f"detection이 있는 프레임: {frames_with_detection}/{frame_count}")
    print(f"결과 동영상: {out_video_path}")
    print("-" * 50)

def main():
    model = YOLO(MODEL_PATH)
    
    # 동영상 파일 찾기
    video_paths = []
    for ext in ("*.mp4", "*.mov", "*.avi", "*.wmv", "*.mkv"):
        video_paths.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))
    video_paths.sort()
    
    if not video_paths:
        print(f"동영상 파일을 찾을 수 없습니다: {VIDEO_DIR}")
        return
    
    print(f"찾은 동영상 파일: {len(video_paths)}개")
    
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for video_path in video_paths:
            process_video(video_path, model, jsonl_file)
    
    print(f"모든 처리 완료!")
    print(f"JSONL 파일: {jsonl_path}")

if __name__ == "__main__":
    main()