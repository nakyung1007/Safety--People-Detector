import numpy as np
from typing import  Optional,Tuple
import cv2

BLUR_KSIZE = 45  # 홀수

def estimate_face_area(keypoints: np.ndarray, box: np.ndarray | list | tuple):
    """
    keypoints: (17,2) or (17,3[x,y,conf])
    box: (x1,y1,x2,y2) - 원본 프레임 좌표
    반환: (x1,y1,x2,y2) or None  (원본 프레임 좌표)
    """
    box = list(map(int, box))
    face_indices = [0, 1, 2, 3, 4]  # nose, eyes, ears
    kpts = keypoints.copy()
    if kpts.ndim == 2 and kpts.shape[1] >= 2:
        face_kpts = kpts[face_indices]
    else:
        return None

    # conf 열이 있으면 0.3 이상만 사용
    if face_kpts.shape[1] >= 3:
        mask = face_kpts[:, 2] > 0.3
        face_kpts = face_kpts[mask][:, :2]

    if len(face_kpts) == 0:
        return None

    x_min, y_min = np.min(face_kpts, axis=0).astype(int)
    x_max, y_max = np.max(face_kpts, axis=0).astype(int)

    # box 범위로 clamp
    x_min, y_min = np.maximum([x_min, y_min], [box[0], box[1]])
    x_max, y_max = np.minimum([x_max, y_max], [box[2], box[3]])
    if x_max <= x_min or y_max <= y_min:
        return None

    # 위/아래 확장
    face_h = y_max - y_min
    expand_up = int(face_h * 3.0)
    expand_down = int(face_h * 3.0)
    face_w = x_max - x_min
    expand_side = int(face_w * 0.2)

    x_min -= expand_side; x_max += expand_side
    y_min -= expand_up;   y_max += expand_down

    # 다시 box로 clamp
    x_min = max(x_min, box[0]); y_min = max(y_min, box[1])
    x_max = min(x_max, box[2]); y_max = min(y_max, box[3])

    if x_max <= x_min or y_max <= y_min:
        return None
    return (int(x_min), int(y_min), int(x_max), int(y_max))

def apply_face_blur(frame: np.ndarray, face_area: Optional[Tuple[int,int,int,int]]):
    if not face_area:
        return frame
    x1, y1, x2, y2 = map(int, face_area)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    roi = frame[y1:y2, x1:x2]
    k = BLUR_KSIZE if BLUR_KSIZE % 2 == 1 else BLUR_KSIZE + 1
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    frame[y1:y2, x1:x2] = blurred
    return frame
