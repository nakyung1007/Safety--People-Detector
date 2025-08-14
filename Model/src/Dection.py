# pose_filter_and_save.py
# pip install ultralytics opencv-python

import os, json, glob
import cv2
from ultralytics import YOLO

# --------------------- 설정 ---------------------
MODEL_PATH = "../utils/yolo11n-pose.pt"  # 또는 "yolo11n-pose.pt"
IMG_DIR    = "../../Data/Sample/train"            # 이미지 폴더
OUT_DIR    = "../../Data/outputs"                 # 결과 저장 폴더
IMG_SIZE   = 640                                  # 추론 해상도
DET_CONF   = 0.25                                 # 박스 confidence 임계값
KPT_CONF   = 0.3                                  # 발관련 keypoint confidence 임계값
DRAW_KPTS  = True                                 # keypoint도 함께 그림
# ------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
jsonl_path = os.path.join(OUT_DIR, "detections.jsonl")

# COCO 17 기준 인덱스 (Ultralytics pose 기본)
# 15: left_ankle, 16: right_ankle (발목)
LEFT_ANKLE_IDX  = 15
RIGHT_ANKLE_IDX = 16

def feet_visible(kpt_xy, kpt_conf, thr=KPT_CONF):
    """
    kpt_xy: (K, 2), kpt_conf: (K,) or None
    발목(ankle) keypoint가 둘 다 threshold 이상인지 확인
    """
    if kpt_xy is None or kpt_conf is None:
        return False
    K = kpt_xy.shape[0]
    if K <= max(LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX):
        # 17개보다 적으면 발목이 없음
        return False
    la_ok = kpt_conf[LEFT_ANKLE_IDX]  is not None and kpt_conf[LEFT_ANKLE_IDX]  >= thr
    ra_ok = kpt_conf[RIGHT_ANKLE_IDX] is not None and kpt_conf[RIGHT_ANKLE_IDX] >= thr
    return la_ok or ra_ok #둘다 보일때만 True

def draw_person(frame, xyxy, kpt_xy=None, kpt_conf=None):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if DRAW_KPTS and kpt_xy is not None:
        for i, (x, y) in enumerate(kpt_xy):
            if kpt_conf is None or kpt_conf[i] is None or kpt_conf[i] < KPT_CONF:
                continue
            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)
        # 발목은 다른 컬러로 강조
        if kpt_xy.shape[0] > RIGHT_ANKLE_IDX:
            for idx in (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX):
                if kpt_conf is not None and kpt_conf[idx] is not None and kpt_conf[idx] >= KPT_CONF:
                    cv2.circle(frame, (int(kpt_xy[idx,0]), int(kpt_xy[idx,1])), 5, (0, 0, 255), -1)

def main():
    model = YOLO(MODEL_PATH)

    img_paths = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
        img_paths.extend(glob.glob(os.path.join(IMG_DIR, ext)))
    img_paths.sort()

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for p in img_paths:
            res = model(p, imgsz=IMG_SIZE, conf=DET_CONF, verbose=False)
            r = res[0]

            # Ultralytics result tensors
            boxes   = r.boxes
            kpts    = r.keypoints  # .xy [N,K,2], .conf [N,K]
            img_bgr = cv2.imread(p)

            if boxes is None or len(boxes) == 0:
                continue

            xyxys = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            # pose keypoints
            kp_xy   = kpts.xy.cpu().numpy() if kpts is not None and kpts.xy is not None else None  # (N,K,2)
            kp_conf = kpts.conf.cpu().numpy() if kpts is not None and kpts.conf is not None else None  # (N,K)

            kept = 0
            for i, xyxy in enumerate(xyxys):
                this_kp_xy   = kp_xy[i]   if kp_xy is not None   and i < kp_xy.shape[0]   else None
                this_kp_conf = kp_conf[i] if kp_conf is not None and i < kp_conf.shape[0] else None

                # 발목(=발끝 근사) 둘 다 확실히 잡힌 사람만 유지
                if not feet_visible(this_kp_xy, this_kp_conf, KPT_CONF):
                    continue

                kept += 1
                draw_person(img_bgr, xyxy, this_kp_xy, this_kp_conf)

                # JSONL 한 줄 저장
                row = {
                    "image_path": os.path.abspath(p),
                    "bbox_xyxy": [float(x) for x in xyxy.tolist()],
                    "score": float(scores[i]),
                    "kpts_xy": None if this_kp_xy is None else [[float(a), float(b)] for a,b in this_kp_xy.tolist()],
                    "kpts_conf": None if this_kp_conf is None else [None if c is None else float(c) for c in this_kp_conf.tolist()],
                }
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")

            # 유지된 객체가 있을 때만 시각화 저장
            if kept > 0:
                out_path = os.path.join(OUT_DIR, "images", os.path.basename(p))
                cv2.imwrite(out_path, img_bgr)

if __name__ == "__main__":
    main()
