from typing import Optional, Dict
import numpy as np
from ultralytics import YOLO
import cv2

NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_ANKLE, R_ANKLE = 15, 16

def get_person_part(kpt_xy, kpt_conf, conf_thresh):
    """사람의 어느 부분이 탐지되었는지 확인"""
    if kpt_xy is None or kpt_conf is None:
        return "unknown"
    
    # COCO keypoints 기준
    face_points = [0,1,2,3,4]  # nose, eyes, ears
    upper_body = [5,6,7,8,9,10]  # shoulders, elbows, wrists
    lower_body = [11,12,13,14,15,16]  # hips, knees, ankles
    
    face_visible = sum(1 for i in face_points if kpt_conf[i] > conf_thresh) >= 2
    upper_visible = sum(1 for i in upper_body if kpt_conf[i] > conf_thresh) >= 3
    lower_visible = sum(1 for i in lower_body if kpt_conf[i] > conf_thresh) >= 3
    
    if face_visible and upper_visible and lower_visible:
        return "full_body"
    elif face_visible and upper_visible:
        return "upper_body_with_face"
    elif upper_visible:
        return "upper_body"
    elif lower_visible:
        return "lower_body"
    elif face_visible:
        return "face"
    else:
        return "partial"
    
class YOLOPeopleDetector:
    def __init__(self, model: YOLO, img_size: int = 640, det_conf: float = 0.25, device: str = "cpu"):
        self.model = model
        self.img_size = img_size
        self.det_conf = det_conf
        self.device = device

    def detect_frame(self, frame) -> Dict[str, Optional[np.ndarray]]:
        r = self.model.predict(
            frame,
            imgsz=self.img_size,
            conf=self.det_conf,
            classes=[0],     # person만
            verbose=False,
            device=self.device
        )[0]

        boxes, kpts = r.boxes, r.keypoints

        if boxes is not None and len(boxes) > 0:
            xyxys  = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
        else:
            xyxys  = np.empty((0, 4), dtype=float)
            scores = np.empty((0,), dtype=float)

        kpt_xy, kpt_conf = None, None
        if kpts is not None:
            if kpts.xy is not None:
                kpt_xy = kpts.xy.cpu().numpy()
            if getattr(kpts, "conf", None) is not None:
                kpt_conf = kpts.conf.cpu().numpy()

        # 결과에 person_part 정보 추가
        person_parts = []
        if kpt_xy is not None and kpt_conf is not None:
            for i in range(len(xyxys)):
                this_kpt_xy = kpt_xy[i] if i < kpt_xy.shape[0] else None
                this_kpt_conf = kpt_conf[i] if i < kpt_conf.shape[0] else None
                part = get_person_part(this_kpt_xy, this_kpt_conf, self.det_conf)
                person_parts.append(part)

        return {
            "kpt_xy": kpt_xy,
            "kpt_conf": kpt_conf,
            "person_parts": person_parts
        }
        
def person_visible(kpt_xy, kpt_conf, thr: float = 0.30) -> bool:
    if kpt_xy is None or kpt_conf is None:
        return False
    K = kpt_xy.shape[0]
    if K <= max(R_EYE, R_SHOULDER, R_ANKLE):
        return False

    def ok(i): 
        c = kpt_conf[i]
        return (c is not None) and (float(c) >= thr)

    face_ok = ok(NOSE) or ok(L_EYE) or ok(R_EYE)
    shoulder_ok = ok(L_SHOULDER) or ok(R_SHOULDER)
    ankle_ok = ok(L_ANKLE) or ok(R_ANKLE)
    return face_ok or shoulder_ok or ankle_ok

def draw_person(frame, xyxy, track_id=None, kpt_xy=None, kpt_conf=None,
                kpt_thr: float = 0.30, draw_kpts: bool = True,
                helmet: Optional[bool] = None):
    x1, y1, x2, y2 = map(int, xyxy)
     # 헬멧 착용 상태에 따라 박스 색상 결정
    if helmet is True:
        color = (0, 255, 0) # 헬멧 착용: 초록색
    elif helmet is False:
        color = (0, 0, 255) # 헬멧 미착용: 빨간색
    else: # 헬멧 상태 확인 안됨
        color = (255, 255, 0) # 알 수 없음: 청록색
        
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    
    # 표시할 텍스트 준비
    label_id = f"ID: {int(track_id)}" if track_id is not None else ""
    label_helmet = "" if helmet is None else ("HELMET" if helmet else "NO-HELMET")

    # 텍스트 위치(겹치지 않게 두 줄로)
    y_id = max(12, y1 - 24)   # ID 윗줄
    y_hm = max(12, y1 - 6)    # 헬멧 아랫줄
    
    # 텍스트 그리기
    cv2.putText(frame, label_id, (x1, y_id), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, label_helmet, (x1, y_hm), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if not draw_kpts or kpt_xy is None or kpt_conf is None:
        return
    
    for i, (x, y) in enumerate(kpt_xy):
        if float(kpt_conf[i]) >= kpt_thr:
            cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1)

    # 일반 키포인트(흰색)
    for i, (x, y) in enumerate(kpt_xy):
        ci = kpt_conf[i]
        if ci is None or float(ci) < kpt_thr:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, (255,255,255), -1)

    # 강조 포인트: 얼굴(초록), 어깨(파랑), 발목(빨강)
    def emph(idxs, bgr, r=5):
        for idx in idxs:
            ci = kpt_conf[idx]
            if ci is not None and float(ci) >= kpt_thr:
                x, y = map(int, kpt_xy[idx])
                cv2.circle(frame, (x, y), r, bgr, -1)

    emph([NOSE, L_EYE, R_EYE], (0,255,0))
    emph([L_SHOULDER, R_SHOULDER], (255,0,0))
    emph([L_ANKLE, R_ANKLE], (0,0,255))
    
     # 탐지된 부분 표시
    part = get_person_part(kpt_xy, kpt_conf, kpt_thr)
    cv2.putText(frame, f"Part: {part}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)