# people_detection_eval.py 라는 새로운 파일에 아래 코드를 작성
from ultralytics import YOLO

# config.py 에서 MODEL_PATH를 import
from config import MODEL_PATH

def evaluate_people_detection():
    print("사람 탐지 모델(People Detection) mAP 평가 시작...")
    
    # 1. 모델 로드 (config.py의 MODEL_PATH 사용)
    model = YOLO(MODEL_PATH)

    # 2. 검증 데이터셋 경로 지정
    # ★★★★ 이곳에 정답 라벨이 있는 데이터셋의 .yaml 파일 경로를 입력하세요. ★★★★
    val_dataset_path = 'path/to/your/people_detection_val_dataset.yaml'
    
    # 3. model.val() 함수로 mAP 계산
    # classes=[0]을 지정하여 사람(person) 클래스만 평가합니다.
    metrics = model.val(
        data=val_dataset_path, 
        imgsz=640, 
        conf=0.25, 
        classes=[0],
        save_json=True # 결과를 .json 파일로 저장하여 추후 분석 가능
    )
    
    # 결과 출력
    mAP50 = metrics.box.map50
    mAP50_95 = metrics.box.map
    
    print(f"\nPeople Detection mAP@0.50: {mAP50:.4f}")
    print(f"People Detection mAP@0.50-0.95: {mAP50_95:.4f}")

if __name__ == "__main__":
    evaluate_people_detection()