from typing import List, Tuple, Optional
import os, tempfile, cv2

from inference_sdk import InferenceHTTPClient, InferenceConfiguration


def make_helmet_client(api_url: str, api_key: str,
                       min_conf: float, iou_thresh: float,
                       model_id: Optional[str] = None) -> InferenceHTTPClient:
   
    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    client.configure(InferenceConfiguration(
        confidence_threshold=float(min_conf),
        iou_threshold=float(iou_thresh)
    ))
    if model_id:
        client._model_id = model_id  
    return client


def _ensure_min_short_side(bgr, min_ss: int = 640):
    h, w = bgr.shape[:2]
    ss = min(h, w)
    if ss >= min_ss:
        return bgr, 1.0
    s = float(min_ss) / float(ss)
    resized = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    return resized, s


def _cxcywh_to_xyxy(x, y, w, h):
    return [x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0]


def _clip_xyxy(xyxy, w, h):
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    return [x1, y1, x2, y2]


def _iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def infer_helmet_boxes(client: InferenceHTTPClient,
                       frame_bgr,
                       label_name: str,
                       min_conf: float = 0.65,
                       min_short_side: int = 640) -> List[Tuple[list, float]]:
    
    if not hasattr(client, "_model_id"):
        raise ValueError("helmet_detection: client._model_id가 설정되지 않았습니다. "
                         "make_helmet_client(model_id=...)로 생성하거나, "
                         "infer 호출 전 client._model_id = '<model-id>' 를 지정하세요.")

    H, W = frame_bgr.shape[:2]
    img_for_infer, s = _ensure_min_short_side(frame_bgr, min_short_side)

    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(tmp, img_for_infer)
    try:
        resp = client.infer(tmp, model_id=client._model_id)
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

    preds = resp.get("predictions", []) or []
    out: List[Tuple[list, float]] = []
    ln = (label_name or "").strip().lower()

    for p in preds:
        name = (p.get("class") or p.get("class_name") or "").strip().lower()
        conf = float(p.get("confidence", 0.0))
        if name != ln or conf < float(min_conf):
            continue

        cx = float(p.get("x")); cy = float(p.get("y"))
        pw = float(p.get("width")); ph = float(p.get("height"))
        xyxy = _cxcywh_to_xyxy(cx, cy, pw, ph)

        if s != 1.0:
            xyxy = [v / s for v in xyxy]

        xyxy = _clip_xyxy(xyxy, W, H)
        out.append((xyxy, conf))

    return out


def person_has_helmet(person_xyxy,
                      helmet_boxes: List[Tuple[list, float]],
                      top_ratio: float = 0.45,
                      iou_thr: float = 0.05) -> Tuple[bool, Optional[float]]:
    
    x1, y1, x2, y2 = person_xyxy
    best = 0.0
    found = False
    for hxyxy, score in helmet_boxes:
        if _iou_xyxy(person_xyxy, hxyxy) < iou_thr:
            continue
        hx = 0.5 * (hxyxy[0] + hxyxy[2])
        hy = 0.5 * (hxyxy[1] + hxyxy[3])
        in_box = (x1 <= hx <= x2) and (y1 <= hy <= y2)
        in_top = hy <= (y1 + top_ratio * (y2 - y1))
        if in_box and in_top:
            found = True
            if score > best:
                best = score
    return (found, (best if found else None))
