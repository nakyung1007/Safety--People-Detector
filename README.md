# Safety People Detector 

## Overview
본 프로젝트는 **(주)머제스** 기업과의 기업 연계형으로 진행되었으며,
**조선대학교 센서융합인공지능연구실**의 학부 연구생 과제로 수행되었습니다.</br>
건설 현장에서 사람 전신 탐지와 PPE(개인 보호구: 안전모, 안전조끼) 착용 여부 감지를 수행하는 것으로 CPU에서 약 15 FPS에서 작동되게 하였습니다.


## Contributors
<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/nakyung1007"><img height="110px" src="https://avatars.githubusercontent.com/u/126228131?v=4"/></a>
            <br />
            <a href="https://github.com/nakyung1007"><strong>조나경</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/iooah"><img height="110px"  src="https://avatars.githubusercontent.com/u/144919371?v=4"/></a>
              <br />
              <a href="https://github.com/iooah"><strong>김수아</strong></a>
              <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/wlalslzzang"><img height="110px"  src="https://avatars.githubusercontent.com/u/189085901?v=4"/></a>
              <br />
              <a href="https://github.com/wlalslzzang"><strong>정지민</strong></a>
              <br />
        </td>
    </tr>
</table>  

지도교수: 김원열 교수 (조선대학교 - 센서융합인공지능연구실)

산학협력 기업: (주)머제스


## 📊 Results

<table>
<tr>
<td>

###  Performance (Speed)
| Metric | Value |
|--------|-------|
| FPS    | 16.48 FPS |

</td>
<td>

###   Helmet Detection
| Metric    | Value   |
|-----------|---------|
| Accuracy  | 69.09 % |
| F1-Score  | 76.5 %  |
| Recall    | 71.07 % |

</td>
<td>

###  Vest Detection
| Metric    | Value   |
|-----------|---------|
| Accuracy  | 89.87 % |
| F1-Score  | 98.65 % |
| Recall    | 80.97 % |

</td>
<td>

### Bodypart Detection
| Metric    | Value   |
|-----------|---------|
| Accuracy  | 91.58 % |
| F1-Score  | 91.58 % |
| Recall    | 95.61 % |

</td>
</tr>
</table>

##  Setup (How to learn)

### 1. Settings
```bash
git clone https://github.com/username/Safety-People-Detector.git
cd Safety-People-Detector

# 패키지 설치
pip install -r requirements.txt
```
### 2. Run the application
```bash
# 단일 영상(video) 실행
python src/yolo/main.py --video/data/test_video.mp4 --device cpu

# 폴더 내 영상(video) 실행
python src/yolo/main.py --video_dir data/videos --device cpu
```

##  Model & Environment Setup

| Category   | Details |
|------------|---------|
| Detection Models | YOLO v11 Pose (사람 탐지, 전신/상체/하체 분류) <br> YOLO v11 Detection (Helmet, Vest 탐지) |
| Tracking   | ByteTrack  <br> IoU 기반 사람-PPE 매칭 |
| Optimization | 입력 크기 640×640 <br> 15 프레임 단위 추론 |



## Pipeline
1. Data Input → CCTV/Camera  
2. Preprocessing → Normalization (640×640)  
3. Detection → YOLO Pose (person), YOLO Detection (helmet/vest)  
4. Tracking → ByteTrack, IoU Matching  
5. Post-processing → Face Blur  
6. Output → Visualization (video), CSV logs
