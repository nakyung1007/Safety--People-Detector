from typing import Optional, Dict
import numpy as np
from ultralytics import YOLO
import cv2

NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_ANKLE, R_ANKLE = 15, 16

class PeopleDetector:
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

        return {"xyxys": xyxys, "scores": scores, "kpt_xy": kpt_xy, "kpt_conf": kpt_conf}

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
                kpt_thr: float = 0.30, draw_kpts: bool = True):
    x1, y1, x2, y2 = map(int, xyxy)
    color = (0, 255, 0) if track_id is None else [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)][int(track_id) % 6]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    if track_id is not None:
        cv2.putText(frame, f"ID:{int(track_id)}", (x1, max(10, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if not draw_kpts or kpt_xy is None or kpt_conf is None:
        return

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
                x, y = map(int, kpt_xy[idx]); cv2.circle(frame, (x, y), r, bgr, -1)

    emph([NOSE, L_EYE, R_EYE], (0,255,0))
    emph([L_SHOULDER, R_SHOULDER], (255,0,0))
    emph([L_ANKLE, R_ANKLE], (0,0,255))