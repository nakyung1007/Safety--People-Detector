#데이터 경로
VIDEO_DIR  = "../../Data/Real"           
OUT_DIR    = "../../Data/output"  

#모델 경로
MODEL_PATH = "../model/checkpoints/yolo11n-pose.pt"  
TRACKER_YAML = "bytetrack.yaml"  

# ---- 멀티클래스 PPE 단일 API 설정 ----
PPE_CONFIG = {
    # Roboflow Serverless Inference 엔드포인트/키/모델
    "api_url": "https://serverless.roboflow.com",
    "api_key": "7iUzwZnQQnmfd4Mqm9tg",          # 바꾸기
    "model_id": "construction-site-safety/27",    # 바꾸기 예: "construction-ppe-multi/4"

    # 서버측 임계값
    "conf_thresh": 0.25,
    "iou_thresh": 0.5,

    # 클라이언트 후처리/샘플링 파라미터 (quick_run.py에서 사용)
    "min_short_side": 640,       # API 호출 전 리사이즈 최소 짧은 변
    "post_conf_thr": 0.30,       # 클라에서 한 번 더 confidence 필터

    # 헬멧 매칭용 (사람 박스 상단 몇 %를 머리 영역으로 볼지)
    "helmet_top_ratio": 0.60,
    "helmet_cover_frac": 0.35,
    "helmet_eye_margin": 0.03,

    # 조끼 후처리(색/형상/몸통위치 필터)
    "vest_ratio_thr": 0.008,     # 형광(주황/노랑/라임) 픽셀 비율 임계
    "vest_area_min_frac": 0.0015,
    "vest_area_max_frac": 0.25,
    "vest_aspect_min": 0.35,
    "vest_aspect_max": 3.0,
    "vest_torso_low": 0.15,      # 사람 박스 높이 대비 몸통 하한 비율
    "vest_torso_high": 0.90,     # 사람 박스 높이 대비 몸통 상한 비율
    "vest_color_mode": "all",

    # 호출 주기/상태 유지
    "every": 30,                 # n프레임마다 단일 PPE API 호출
    "carry_ttl": 90              # 이전 결과 carry(프레임 수)
}

'''

VEST_CONFIG = {
    "api_url": "https://serverless.roboflow.com",
    "api_key" : "7iUzwZnQQnmfd4Mqm9tg",
    "model_id" : "construction-ppe-rdhzo/3",
    "conf_thresh": 0.25,                        
    "iou_thresh": 0.5,                          # 서버 IoU
    "label_name": "Vest",                       # ""(빈문자)면 라벨 제한 해제

    # 아래는 선택(후처리/샘플링 파라미터) – 있으면 quick_run이 활용
    "min_short_side": 640,
    "post_conf_thr": 0.35,                      # 클라이언트에서 더 타이트하게 필터
    "ratio_thr": 0.008,                          # 형광색 비율 임계
    "area_min_frac": 0.0015,
    "area_max_frac": 0.25,
    "aspect_min": 0.35,
    "aspect_max": 3.0,
    "torso_low": 0.15,
    "torso_high": 0.90,
    "every": 15,                                # n프레임마다 조끼 API 호출
    "carry_ttl": 90
}

IMG_SIZE   = 640                                  
DET_CONF   = 0.25                                 
KPT_CONF   = 0.3                                 
DRAW_KPTS  = True   
DEVICE     = "cpu"               

HELMET_CONFIG = {
    "api_url": "https://serverless.roboflow.com",
    "api_key": "anGJZZC1W9kgvq6R3Uks",
    "model_id": "industry_safety/2",
    "conf_thresh": 0.20,
    "iou_thresh": 0.5,
    "label_name": "Helmet"                 # 대시보드에서 보이는 정확한 클래스명
}
'''