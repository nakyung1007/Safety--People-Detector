from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from ultralytics import YOLO
import ultralytics

class Tracker:
    def __init__(
        self,
        model: YOLO,
        tracker_name: str = "bytetrack.yaml",  
        img_size: int = 640,
        det_conf: float = 0.25,
        device: str = "cpu",
    ):
        self.model = model
        self.img_size = img_size
        self.det_conf = det_conf
        self.device = device
        self.tracker_yaml = self._resolve_tracker_yaml(tracker_name)
        
         # compact id state
        self._id_map: Dict[int, int] = {}  # orig_id -> compact_id
        self._next_cid: int = 1
        
    def reset_id_mapping(self):
        """동영상 시작 전에 호출해서 compact id를 1부터 다시 시작."""
        self._id_map.clear()
        self._next_cid = 1

    @staticmethod
    def _resolve_tracker_yaml(name: str) -> str:
        base = Path(ultralytics.__file__).resolve().parent
        p = base / "cfg" / "trackers" / name
        return str(p)
    
    def _to_compact_ids(self, orig_ids: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if orig_ids is None:
            return None
        compact = np.empty_like(orig_ids)
        for i, oid in enumerate(orig_ids):
            oi = int(oid)
            if oi < 0:          # 미할당/음수 ID는 -1 유지
                compact[i] = -1
                continue
            if oi not in self._id_map:
                self._id_map[oi] = self._next_cid
                self._next_cid += 1
            compact[i] = self._id_map[oi]
        return compact

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

        orig_ids = None
        if boxes is not None and boxes.id is not None:
            orig_ids = boxes.id.detach().cpu().numpy().astype(int)

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        
        ids = self._to_compact_ids(orig_ids)

        kpt_xy = None
        kpt_conf = None
        if kpts is not None:
            if getattr(kpts, "xy", None) is not None:
                kpt_xy = kpts.xy.detach().cpu().numpy()
            if getattr(kpts, "conf", None) is not None:
                kpt_conf = kpts.conf.detach().cpu().numpy()

        return {
            "xyxys": xyxys,
            "scores": scores,
            "ids": ids,
            "kpt_xy": kpt_xy,
            "kpt_conf": kpt_conf
        }
