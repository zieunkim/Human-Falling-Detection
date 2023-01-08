# Human-Falling-Detection


YOLOv5를 이용한 낙상 감지 AI CCTV 프로그램 

- AI Hub  데이터셋 수집 및 직접 라벨링
<img width="588" alt="image" src="https://user-images.githubusercontent.com/81521991/211191194-c620cffa-d3c6-4aa9-bf00-72ebb3c1dc2d.png">


- 프로그램 실행 시 낙상 판단

1. 낙상 검출 알고리즘

     ㅇ 넘어지는 속력을 이용한다.
     
     ㅇ 낙상 행동에서 땅에 닿기까지 상체는 약 0.8초, 하체는 약 0.5초의 시간이 걸린다고 분석하였다.
     
     ㅇ 사람의 같은 신체 부위의 이용해 속력을 구하고, 그를 통해 낙상을 판단한다.
     
     ㅇ 해당 알고리즘의 장점으로는 낙상 검출의 정확도를 높여준다.

2. 낙상 방향 판단 알고리즘
    
    <img width="375" alt="image" src="https://user-images.githubusercontent.com/81521991/211191751-a5146270-2477-4fd9-ad5e-3afb4ac50c1f.png">

     ㅇ 낙상의 형태를 전방낙상 (그림 a), 후방낙상 (그림 b), 측면낙상 (그림 c, d)으로 구분한다.
     
     ㅇ 객체 너비에 해당하는 Xmin값과 Xmax값의 변화와, 객체 높이에 해당하는 Ymin값과 Ymax값의 변화에 따라 알고리즘 구성
     
     ㅇ 해당 알고리즘의 장점으로는 사람의 신체 특징에 따라, 낙상하였을 때의 신체 위치 변화를 이용하기 때문에 정확도가 높다.
     
     
- 기술 개발 결과 예시
<img width="538" alt="image" src="https://user-images.githubusercontent.com/81521991/211191775-0a7663a4-3c4e-472a-a6ba-033088742a9e.png">


- 참고 문헌


ㅇ YOLOv3 알고리즘을 이용한 실시간 낙상 검출(김지민 외 4)

ㅇ 오픈소스 하드웨어와 RGB 카메라를 이용한 낙상 검출 시스템(황세현 외 1)
