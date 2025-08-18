import json

def evaluate_helmet_accuracy(predictions_file: str, ground_truth_file: str):
    """
    JSONL 예측 파일과 정답 JSON 파일을 비교하여 헬멧 감지 정확도를 계산합니다.
    """
    print("헬멧 감지 정확도(Accuracy) 평가 시작...")

    # 1. 정답 데이터 로드
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    # 2. 예측 결과와 정답 데이터 비교
    correct_count = 0
    total_count = 0
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            pred = json.loads(line)
            
            # 예측 결과에서 필요한 정보 추출
            video_path = pred["video_path"]
            frame_num = str(pred["frame_number"])
            track_id = str(pred["track_id"])
            predicted_helmet = pred["helmet"]
            
            # 예측이 "Skip"인 경우는 평가에서 제외
            if predicted_helmet == "Skip":
                continue

            # 정답 데이터에서 해당 정보 찾기
            # gt_data 형식: { "video_path": { "frame_num": { "track_id": True/False } } }
            gt_video = gt_data.get(video_path, {})
            gt_frame = gt_video.get(frame_num, {})
            gt_helmet_state = gt_frame.get(track_id)

            if gt_helmet_state is not None:
                total_count += 1
                
                # 예측 결과 ('True'/'False')를 boolean으로 변환
                predicted_bool = predicted_helmet == "True"
                
                if predicted_bool == gt_helmet_state:
                    correct_count += 1

    if total_count > 0:
        accuracy = correct_count / total_count
        print(f"총 평가 객체: {total_count}개")
        print(f"정확한 예측: {correct_count}개")
        print(f"최종 헬멧 감지 정확도: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    else:
        print("평가할 데이터가 없습니다. 예측 및 정답 파일을 확인해주세요.")

if __name__ == "__main__":
    # 이 경로들을 실제 파일 경로로 수정해야 합니다.
    # ground_truth_file은 직접 라벨링하여 만들어야 합니다.
    evaluate_helmet_accuracy(
        predictions_file="path/to/your/detections.jsonl",
        ground_truth_file="path/to/your/helmet_ground_truth.json"
    )