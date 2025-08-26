import os,sys
from typing import List, Dict, Optional
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import *

class PPEDetector:
    def __init__(self, model_path, conf=0.25, device='cpu', imgsz=640, cache_size=5):
        self.model = YOLO(model_path)
        self.model.cpu()  # 명시적으로 CPU 모드로 설정
        self.conf = conf
        self.device = 'cpu'  # 항상 CPU 사용
        self.imgsz = imgsz
        self.cache_size = cache_size
        self.result_cache = {}  # frame_idx -> result
        
        # CPU 성능 최적화를 위한 설정
        self.model.model.half = False  # FP16 비활성화
        self.model.model.fuse()  # 모델 퓨징
        
    def infer(self, frame, frame_idx=None, batch_size=1):
        """
        프레임 또는 프레임 배치에 대해 PPE 탐지 수행
        
        Args:
            frame: 단일 프레임 또는 프레임 배치
            frame_idx: 현재 프레임 인덱스 (캐시 사용 시 필요)
            batch_size: 배치 크기
        """
        # 캐시된 결과가 있으면 반환
        if frame_idx is not None and frame_idx in self.result_cache:
            return self.result_cache[frame_idx]
        
        # 배치 처리
        results = self.model(frame, 
                           imgsz=self.imgsz, 
                           conf=self.conf, 
                           device=self.device, 
                           verbose=False,
                           batch=batch_size)
        
        # 단일 프레임인 경우
        if not isinstance(frame, (list, tuple)):
            result = results[0]
            # 캐시 업데이트
            if frame_idx is not None:
                self.result_cache[frame_idx] = result
                # 캐시 크기 제한
                if len(self.result_cache) > self.cache_size:
                    min_idx = min(self.result_cache.keys())
                    del self.result_cache[min_idx]
            return result
        
        # 배치 처리 결과 반환
        return results

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)

def _center_inside(inner: np.ndarray, outer: np.ndarray) -> bool:
    cx = (inner[0] + inner[2]) * 0.5
    cy = (inner[1] + inner[3]) * 0.5
    return (outer[0] <= cx <= outer[2]) and (outer[1] <= cy <= outer[3])

def assign_ppe_to_person_boxes(ppe_res,
                               person_xyxys: np.ndarray,
                               iou_thr: float = 0.05,
                               prev_results: List[Dict[str, Optional[bool]]] = None,
                               confidence_threshold: float = 0.5) -> List[Dict[str, Optional[bool]]]:
    """
    ppe_res: PPEDetector.infer(frame) 결과(한 프레임)
    person_xyxys: (N,4) array
    prev_results: 이전 프레임의 결과 (temporal smoothing용)
    confidence_threshold: PPE 탐지 신뢰도 임계값
    반환: [{"helmet": T/F/None, "vest": T/F/None}, ...] 길이 N
    규칙: 
    1. nohat/novest가 잡히면 False 우선
    2. hat/vest가 잡히면 True
    3. 이전 결과가 있으면 temporal smoothing 적용
    4. 그 외의 경우 None
    매칭: PPE 박스 중심점이 person 내부 or IoU >= iou_thr
    """
    if ppe_res is None or ppe_res.boxes is None or len(ppe_res.boxes) == 0 or len(person_xyxys) == 0:
        return [{"helmet": None, "vest": None} for _ in range(len(person_xyxys))]

    names = ppe_res.names  # 예: {0:'hat',1:'nohat',2:'novest',3:'person',4:'vest'}
    clss = ppe_res.boxes.cls.cpu().numpy().astype(int)
    xyxy = ppe_res.boxes.xyxy.cpu().numpy()

    idx_hat    = [i for i,c in enumerate(clss) if names[c] == "hat"]
    idx_nohat  = [i for i,c in enumerate(clss) if names[c] == "nohat"]
    idx_vest   = [i for i,c in enumerate(clss) if names[c] == "vest"]
    idx_novest = [i for i,c in enumerate(clss) if names[c] == "novest"]

    confs = ppe_res.boxes.conf.cpu().numpy()  # confidence scores 추가
    
    out = []
    for p in person_xyxys:
        # 헬멧 검출
        helmet_matches = [(i, _iou(xyxy[i], p)) for i in idx_hat if _center_inside(xyxy[i], p) or _iou(xyxy[i], p) >= iou_thr]
        helmet_false = any(_center_inside(xyxy[i], p) or _iou(xyxy[i], p) >= iou_thr for i in idx_nohat)
        
        # 조끼 검출
        vest_matches = [(i, _iou(xyxy[i], p)) for i in idx_vest if _center_inside(xyxy[i], p) or _iou(xyxy[i], p) >= iou_thr]
        vest_false = any(_center_inside(xyxy[i], p) or _iou(xyxy[i], p) >= iou_thr for i in idx_novest)

        # 최적의 매칭 찾기
        best_helmet = max(helmet_matches, key=lambda x: x[1], default=None)
        best_vest = max(vest_matches, key=lambda x: x[1], default=None)

        # 결과 저장
        result = {
            "helmet": False if helmet_false else (True if best_helmet else None),
            "vest": False if vest_false else (True if best_vest else None),
            "helmet_bbox": xyxy[best_helmet[0]].tolist() if best_helmet else None,
            "helmet_score": float(confs[best_helmet[0]]) if best_helmet else None,
            "vest_bbox": xyxy[best_vest[0]].tolist() if best_vest else None,
            "vest_score": float(confs[best_vest[0]]) if best_vest else None
        }
        out.append(result)
    return out
