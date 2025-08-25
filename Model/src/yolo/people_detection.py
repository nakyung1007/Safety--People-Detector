from typing import Optional, Dict, Tuple
import numpy as np
from ultralytics import YOLO
import cv2

NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_ANKLE, R_ANKLE = 15, 16

def get_person_part(kpt_xy, kpt_conf, conf_thresh):
    if kpt_xy is None or kpt_conf is None:
        return "unknown"
    
    # keypoints 기준
    face_points = [0,1,2,3,4]  
    upper_body = [5,6,7,8,9,10] 
    lower_body = [11,12,13,14,15,16] 
    
    face_visible = sum(1 for i in face_points if kpt_conf[i] > conf_thresh) >= 2
    upper_visible = sum(1 for i in upper_body if kpt_conf[i] > conf_thresh) >= 3
    lower_visible = sum(1 for i in lower_body if kpt_conf[i] > conf_thresh) >= 2
    
    if face_visible and upper_visible and lower_visible:
        return "full_body"
    elif upper_visible:
        return "upper_body"
    elif lower_visible:
        return "lower_body"
    elif face_visible:
        return "face"
    else:
        return "None"
    
def _ppe_color_and_text(helmet: Optional[bool], vest: Optional[bool]):
    if helmet is None or vest is None:
        return (255, 255, 0), "unknown"
    if helmet and vest:
        return (0, 255, 0), "helmet+vest"
    if (helmet and not vest):
        return (0, 165, 255), "helmet"
    if (not helmet and vest):
        return (0, 165, 255), "vest"
    return (0, 0, 255), "none"

def draw_person(frame, xyxy, track_id=None, kpt_xy=None, kpt_conf=None,
                kpt_thr: float = 0.30, draw_kpts: bool = False,
                helmet: Optional[bool] = None, vest: Optional[bool] = None):
    x1, y1, x2, y2 = map(int, xyxy)
    color, ppe_text = _ppe_color_and_text(helmet, vest)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label_id = f"ID: {int(track_id)}" if track_id is not None else ""
    y_id  = max(12, y1 - 24)
    y_ppe = max(12, y1 - 6)
    cv2.putText(frame, label_id, (x1, y_id),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, ppe_text, (x1, y_ppe), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    part = get_person_part(kpt_xy, kpt_conf, kpt_thr)
    cv2.putText(frame, f"Part: {part}", (x1, min(y1 + 18, frame.shape[0]-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)