# eval_mota.py 파일을 새로 만들고 아래 코드 작성
import motmetrics as mm
import os

def evaluate_mota(gt_file: str, tracker_results_file: str):
    """
    MOT Challenge 포맷의 정답 파일과 추적 결과 파일을 비교하여 MOTA를 계산합니다.
    """
    print("트래킹(Tracking) MOTA 평가 시작...")
    
    if not os.path.exists(gt_file) or not os.path.exists(tracker_results_file):
        print("파일 경로를 확인해주세요. 평가를 진행할 수 없습니다.")
        return

    # motmetrics 누적 객체 생성
    acc = mm.MOTAccumulator(auto_id=True)
    
    # 정답 파일과 추적 결과 파일 로드
    gt_data = mm.io.loadtxt(gt_file, fmt='mot15-2D')
    tracker_data = mm.io.loadtxt(tracker_results_file, fmt='mot15-2D')
    
    # 프레임별로 데이터 비교
    for frameid, gt_frame in gt_data.groupby(level='FrameId'):
        if frameid not in tracker_data.index:
            continue
            
        tracker_frame = tracker_data.loc[frameid]
        
        # IoU 기반 매칭 거리 행렬 계산
        dists = mm.distances.iou_matrix(
            gt_frame[['X', 'Y', 'Width', 'Height']],
            tracker_frame[['X', 'Y', 'Width', 'Height']],
            max_iou=0.5 # IoU 임계값
        )
        
        # 누적 객체 업데이트
        acc.update(
            gt_frame['Id'].values,
            tracker_frame['Id'].values,
            dists
        )
    
    # 최종 지표 계산 및 출력
    mh = mm.metrics.create()
    summary = mh.compute(
        acc, 
        metrics=['mota', 'motp', 'num_false_positives', 'num_misses', 'num_id_switches'], 
        name='Overall'
    )
    
    print("\nMOTA 계산 결과:")
    print(summary)
    
    print(f"\n최종 MOTA: {summary['mota'].iloc[0]:.4f}")

if __name__ == "__main__":
    # 이 경로들을 실제 파일 경로로 수정해야 합니다.
    evaluate_mota(
        gt_file="path/to/your/mot_ground_truth.txt",
        tracker_results_file="path/to/your/tracking_mot/video_name.txt"
    )