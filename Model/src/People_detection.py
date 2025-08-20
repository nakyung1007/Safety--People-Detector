# people_detection_copy.py
from typing import Optional, Dict, Tuple
import numpy as np
import cv2

try:
    # YOLOPeopleDetector를 쓸 수도 있으니 예외 처리
    from ultralytics import YOLO
except Exception:
    YOLO = None  # 사용 안 하면 무시

# --- COCO keypoint index ---
NOSE, L_EYE, R_EYE = 0, 1, 2
L_SHOULDER, R_SHOULDER = 5, 6
L_ANKLE, R_ANKLE = 15, 16


def get_person_part(kpt_xy, kpt_conf, conf_thresh: float):
    """사람의 어느 부분이 탐지되었는지 요약 라벨 생성."""
    if kpt_xy is None or kpt_conf is None:
        return "unknown"

    face_points  = [0, 1, 2, 3, 4]          # nose, eyes, ears
    upper_body   = [5, 6, 7, 8, 9, 10]      # shoulders, elbows, wrists
    lower_body   = [11, 12, 13, 14, 15, 16] # hips, knees, ankles

    face_visible  = sum(1 for i in face_points  if kpt_conf[i] > conf_thresh) >= 2
    upper_visible = sum(1 for i in upper_body   if kpt_conf[i] > conf_thresh) >= 3
    lower_visible = sum(1 for i in lower_body   if kpt_conf[i] > conf_thresh) >= 2

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
    """선택 사항: 단일 프레임에서 사람+키포인트만 뽑고 싶을 때 사용."""
    def __init__(self, model: "YOLO", img_size: int = 640, det_conf: float = 0.25, device: str = "cpu"):
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
    """키포인트 신뢰도로 사람 존재성(보임)을 빠르게 판정."""
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


# ----- 라벨 유틸 -----
def _stack_label(img, x, y, text, bg_bgr, scale=0.6, thickness=2):
    """라벨을 위에서 아래로 겹치지 않게 차곡차곡 쌓아 올리는 유틸."""
    if not text:
        return y
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    y1 = max(0, y - th - 6)
    cv2.rectangle(img, (x, y1), (x + tw + 6, y), bg_bgr, -1)
    cv2.putText(img, text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)
    return y1 - 4  # 다음 줄은 좀 더 위로


def draw_person(frame,
                xyxy,
                track_id: Optional[int] = None,
                kpt_xy=None,
                kpt_conf=None,
                kpt_thr: float = 0.30,
                draw_kpts: bool = True,
                helmet: Optional[bool] = None,
                vest: Optional[bool] = None,
                part_text: Optional[str] = None,
                vest_box: Optional[Tuple[int, int, int, int]] = None):
    """
    - 사람 박스는 항상 그림.
    - helmet/vest 상태에 따라 'HELMET/NO-HELMET', 'VEST/NO-VEST' 라벨을 표시.
    - 색상 규칙(안전 우선):
        * (helmet == False) or (vest == False)  -> 빨강
        * else if (helmet == True) or (vest == True) -> 초록
        * else (둘 다 None) -> 노랑
    - vest_box가 있으면 얇게 조끼 박스도 그림(디버깅/가시화용).
    """
    x1, y1, x2, y2 = map(int, xyxy)

    # ▶ 색상 결정: violation-first
    if (vest is False) or (helmet is False):
        color = (0, 0, 255)      # RED: 하나라도 미착용
    elif (vest is True) or (helmet is True):
        color = (0, 255, 0)      # GREEN: 착용이 하나라도 있고, 미착용은 없음
    else:
        color = (255, 255, 0)    # YELLOW: 둘 다 None(미확정)

    # 사람 박스
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # 라벨 스택 시작 위치(박스 상단 위)
    y_cursor = max(26, y1 - 6)

    # ID
    id_text = f"ID: {int(track_id)}" if track_id is not None else ""
    y_cursor = _stack_label(frame, x1, y_cursor, id_text, (255, 255, 0))

    # Part
    if part_text is None and kpt_xy is not None and kpt_conf is not None:
        part_text = get_person_part(kpt_xy, kpt_conf, kpt_thr)
    y_cursor = _stack_label(frame, x1, y_cursor, (f"Part: {part_text}" if part_text else ""), (200, 255, 255))

    # Helmet 라벨
    if helmet is True:
        y_cursor = _stack_label(frame, x1, y_cursor, "HELMET", (0, 200, 0))
    elif helmet is False:
        y_cursor = _stack_label(frame, x1, y_cursor, "NO-HELMET", (0, 0, 255))
    # None이면 표시 안 함

    # Vest 라벨
    if vest is True:
        y_cursor = _stack_label(frame, x1, y_cursor, "VEST", (0, 200, 0))
        if vest_box is not None:
            vx1, vy1, vx2, vy2 = map(int, vest_box)
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 200, 255), 2)
    elif vest is False:
        y_cursor = _stack_label(frame, x1, y_cursor, "NO-VEST", (0, 0, 255))
    # None이면 표시 안 함

    # 키포인트 그리기
    if draw_kpts and (kpt_xy is not None) and (kpt_conf is not None):
        # 기본 점(흰색)
        for i, (px, py) in enumerate(kpt_xy):
            if float(kpt_conf[i]) >= kpt_thr:
                cv2.circle(frame, (int(px), int(py)), 3, (255, 255, 255), -1)

        # 강조 포인트
        def emph(idxs, bgr, r=5):
            for idx in idxs:
                ci = kpt_conf[idx]
                if ci is not None and float(ci) >= kpt_thr:
                    px, py = map(int, kpt_xy[idx])
                    cv2.circle(frame, (px, py), r, bgr, -1)

        emph([NOSE, L_EYE, R_EYE], (0, 255, 0))   # 얼굴
        emph([L_SHOULDER, R_SHOULDER], (255, 0, 0))  # 어깨
        emph([L_ANKLE, R_ANKLE], (0, 0, 255))     # 발목
