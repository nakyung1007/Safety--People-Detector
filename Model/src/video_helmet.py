import os, cv2, tempfile,sys
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import HELMET_CONFIG

def ensure_min_short_side(bgr, min_ss=640):
    h, w = bgr.shape[:2]
    ss = min(h, w)
    if ss >= min_ss:
        return bgr
    s = float(min_ss) / ss
    return cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)

def main(video_path: str, frame_step: int = 3, min_short_side: int = 640):
    client = InferenceHTTPClient(
        api_url=HELMET_CONFIG["api_url"],
        api_key=HELMET_CONFIG["api_key"]
    )
    client.configure(InferenceConfiguration(
        confidence_threshold=HELMET_CONFIG["conf_thresh"],
        iou_threshold=HELMET_CONFIG["iou_thresh"]
    ))
    model_id   = HELMET_CONFIG["model_id"]
    label_name = str(HELMET_CONFIG["label_name"]).strip()  

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_step != 0:
            continue

        frame_for_infer = ensure_min_short_side(frame, min_short_side)

        fd, tmp = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
        cv2.imwrite(tmp, frame_for_infer)
        try:
            resp = client.infer(tmp, model_id=model_id)
        finally:
            try: os.remove(tmp)
            except: pass

        preds = resp.get("predictions", []) or []
        has_helmet = any(
            (p.get("class") or p.get("class_name") or "").strip() == label_name
            for p in preds
        )
        print(f"frame {frame_idx}: {'HELMET' if has_helmet else 'NO-HELMET'}")

    cap.release()

if __name__ == "__main__":
    main(video_path="/Users/chonakyung/Library/CloudStorage/GoogleDrive-whskrud1007@gmail.com/내 드라이브/학부 연구생_2025/Safety--People-Detector/Data/Video/human-accident_jam_rgb_0957_cctv1.mp4", frame_step=10, min_short_side=640)