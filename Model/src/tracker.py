from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from ultralytics import YOLO
import ultralytics

class Tracker:
    def __init__(
        self,
        model: YOLO,
        tracker_name: str = "bytetrack.yaml", # byteTrack  
        # tracker_name: str = "botsort.yaml", # Bot-track
        img_size: int = 640,
        det_conf: float = 0.25,
        device: str = "cpu",
    ):
        self.model = model
        self.img_size = img_size
        self.det_conf = det_conf
        self.device = device
        self.tracker_yaml = self._resolve_tracker_yaml(tracker_name)

    @staticmethod
    def _resolve_tracker_yaml(name: str) -> str:
        base = Path(ultralytics.__file__).resolve().parent
        p = base / "cfg" / "trackers" / name
        return str(p)

    def track_frame(self, frame) -> Dict[str, Optional[np.ndarray]]:
        results = self.model.track(
            frame,
            imgsz=self.img_size,
            conf=self.det_conf,
            classes=[0],             # person만
            persist=True,            # 트랙 상태 유지
            verbose=False,
            device=self.device,
            tracker=self.tracker_yaml
        )
        r = results[0]

        boxes = r.boxes
        kpts  = r.keypoints

        xyxys = boxes.xyxy.cpu().numpy() if (boxes is not None and len(boxes) > 0) else np.empty((0,4), dtype=float)
        scores = boxes.conf.cpu().numpy() if (boxes is not None and len(boxes) > 0) else np.empty((0,), dtype=float)

        ids = None
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)

        kpt_xy = None
        kpt_conf = None
        if kpts is not None:
            if kpts.xy is not None:
                kpt_xy = kpts.xy.cpu().numpy()
            if getattr(kpts, "conf", None) is not None:
                kpt_conf = kpts.conf.cpu().numpy()

        return {
            "xyxys": xyxys,
            "scores": scores,
            "ids": ids,
            "kpt_xy": kpt_xy,
            "kpt_conf": kpt_conf
        }
