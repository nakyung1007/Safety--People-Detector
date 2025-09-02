from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from ultralytics import YOLO
import ultralytics
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import *

class Tracker:
    def __init__(
        self,
        model: YOLO,
        tracker_name: str = TRACKER_YAML,
        img_size: int = IMG_SIZE,
        det_conf: float = DET_CONF,
        device: str = 'cpu', 
    ):
        self.model = model
        self.model.cpu()  
        self.img_size = img_size
        self.det_conf = det_conf
        self.device = 'cpu'
        self.tracker_yaml = self._resolve_tracker_yaml(tracker_name)
        
        self.model.model.half = False  
        self.model.model.fuse()  
        
        self._id_map: Dict[int, int] = {}  
        self._next_cid: int = 1
        
    def reset_id_mapping(self):
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
            if oi < 0:          
                compact[i] = -1
                continue
            if oi not in self._id_map:
                self._id_map[oi] = self._next_cid
                self._next_cid += 1
            compact[i] = self._id_map[oi]
        return compact

    def track_frame(self, frame) -> Dict[str, Optional[np.ndarray]]:
        """단일 프레임에 대한 추적을 수행"""
        results = self.model.track(
            frame,
            imgsz=self.img_size,
            conf=self.det_conf,
            classes=[0],             
            persist=True,            
            verbose=False,
            device=self.device,
            tracker=self.tracker_yaml
        )
        return self._process_results(results[0])
    
    def track_video(self, video_path: str) -> list[Dict[str, Optional[np.ndarray]]]:
        """비디오 전체에 대한 추적을 수행"""
        results = self.model.track(
            source=video_path,
            imgsz=self.img_size,
            conf=self.det_conf,
            classes=[0],             
            persist=True,           
            verbose=False,
            device=self.device,
            tracker=self.tracker_yaml
        )
        return [self._process_results(r) for r in results]
    

    def track_video_stream(self, video_path: str):
        """비디오 전체에 대한 추적을 수행하고 제너레이터 반환"""
        results_generator = self.model.track(
            source=video_path,
            imgsz=self.img_size,
            conf=self.det_conf,
            classes=[0],             
            persist=True,           
            verbose=False,
            device=self.device,
            tracker=self.tracker_yaml,
            stream=True
        )
        return results_generator
    
    
    def _process_results(self, r) -> Dict[str, Optional[np.ndarray]]:
        """추적 결과 처리"""
        boxes = r.boxes
        kpts = r.keypoints

        xyxys = boxes.xyxy.cpu().numpy() if (boxes is not None and len(boxes) > 0) else np.empty((0,4), dtype=float)
        scores = boxes.conf.cpu().numpy() if (boxes is not None and len(boxes) > 0) else np.empty((0,), dtype=float)

        orig_ids = None
        if boxes is not None and boxes.id is not None:
            orig_ids = boxes.id.detach().cpu().numpy().astype(int)

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
