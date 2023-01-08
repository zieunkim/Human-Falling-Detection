#!/usr/bin/env python
# coding: utf-8

# In[39]:


import cv2
import numpy as np
import time
import math
import sys
import torch
from time import time

video_path = './video/test2.mp4'
srt_path = './video/test2.srt'

# pretrained (사람만 검출 - 0번째 class 번호)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]
# custom 학습
# model = torch.hub.load('.', 'custom', path='./runs/train/exp2/weights/best.pt', source='local')

def get_hmsms(dt):
    ms = int((dt - int(dt))*1000)
    dt = int(dt)
    hh, dt = divmod(dt, 3600)
    mm, ss = divmod(dt, 60)
    return (hh, mm, ss, ms)

# Video input
cap = cv2.VideoCapture(video_path)

if not cap.isOpened:
    print('--(!)Error opening video capture')
    sys.exit(1)

Ghh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
Gww = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
FPS = cap.get(cv2.CAP_PROP_FPS)
NUM = cap.get(cv2.CAP_PROP_FRAME_COUNT)

Ghh, Gww = int(Ghh), int(Gww)

print(f"video_path = {video_path}")
print(f"\nvideo resolution: {[Gww, Ghh]}, fps:{FPS}")

rr = input("start?(y/n):")
if(rr.upper() == 'N'):
    sys.exit(1)

# x, y, w, h = rect
x = int(Gww*0.1)
y = int(Ghh*0.1)
w = int(Gww*0.8)
h = int(Ghh*0.8)    

# 필요 변수 선언
i = 0 # frame count & 저장 리스트 비교용
tmp = 0 # 첫번째 detection 구분
notcnt = 0 # 아무것도 검출이 안되는 상황 대비, 일정 카운트 이상일 시 원본 frame 크기로 변경
case_not = 0 # 검출이 안되다가 갑자기 검출될 시 발생하는 예외 제거
event = 0 # 이벤트 상황 구분
j = 1 # 프레임 별 시간 계산용
check = 1 # 자막 append 간격 지정용
time_tmp = 0 # 시간 cnt 시작
event_cnt = 0 # 이벤트 초기화

#FPS = 15
GDt = 1.0/FPS
t1 = 0.0

# 프레임 간 좌표 계산을 위한 저장용 list 생성, 초기값 append
PX = []
PY = []
X_min = []
X_max = []
Y_min = []
Y_max = []
X_min.append(x)
X_max.append(x+w)
Y_min.append(y)
Y_max.append(y+h)

PX.append((X_max[i] + X_min[i])/2)
PY.append((Y_max[i] + Y_min[i])/2)

file_srt = open(srt_path, 'w')

tt_i = time()

while cap.isOpened():
    tt0 = time()
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break
    
    # 첫번째 detection
    if tmp == 0:
        results = model(img)
        cv2.imshow('Falling Detection', img)
        
        # 사람이 검출될 시
        if len(results.xyxy[0]) > 0:
            x = int(results.pandas().xyxy[0].xmin[0])
            y = int(results.pandas().xyxy[0].ymin[0])
            w = int(results.pandas().xyxy[0].xmax[0]) - x
            h = int(results.pandas().xyxy[0].ymax[0]) - y
            notcnt = 0
            if case_not > 0:
                case_not-=1

            # 임의로 박스 주변 crop 범위 지정
            x_ = int((2*x-w) / 2)
            y_ = int((2*y-h) / 2)
            w_ = int(2*w)
            h_ = int(2*h)
            
            if x_ <= 0: x_ = 0
            if y_ <= 0: y_ = 0
            
            xw = x_ + w_
            yh = y_ + h_

            if xw >= Gww: 
                xw = Gww
            if yh >= Ghh: 
                yh = Ghh

            ##### 낙상 판단 방향
            X_min.append(x)
            X_max.append(x+w)
            Y_min.append(y)
            Y_max.append(y+h)
        
            tmp += 1
        
        if time_tmp == 0:
            hh1 = 0
            mm1 = 0
            ss1 = 0
            ms1 = 0
            hh2, mm2, ss2, ms2 = get_hmsms(GDt);
            time_tmp+=1
            
        qq = '정상'
        t1 = t1 + GDt
        hh1, mm1, ss1, ms1 = get_hmsms(t1)
        hh2, mm2, ss2, ms2 = get_hmsms(t1 + GDt);
        
        print("%d\n%02d:%02d:%02d,%03d --> %02d:%02d:%02d,%03d\n%s\n" % (j, hh1, mm1, ss1, ms1, hh2, mm2, ss2, ms2, qq), file=file_srt)
        j+=1
        
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            break
    
    # crop된 frame 안에서 detection 진행
    if tmp > 0:
        image = img[y_:yh, x_:xw]
        # image = img
        results = model(image)
        
        t1 = t1 + GDt
        hh1, mm1, ss1, ms1 = get_hmsms(t1)
        hh2, mm2, ss2, ms2 = get_hmsms(t1 + GDt);
        
        # 축소된 프레임 안에서의 좌표 설정, 조정된 값 계산
        if len(results.xyxy[0]) > 0:
            x = int(results.pandas().xyxy[0].xmin[0]) + x_
            y = int(results.pandas().xyxy[0].ymin[0]) + y_
            w = int(results.pandas().xyxy[0].xmax[0]) + x_ - x
            h = int(results.pandas().xyxy[0].ymax[0]) + y_ - y
            notcnt = 0
            if case_not > 0:
                case_not-=1
        elif len(results.xyxy[0]) == 0 and i % 10 == 0: # 검출 안될 시 image 크기 조금씩 증가, 오래 검출 안될 시 원본 프레임 크기 복구
            x-=1
            y-=1
            w+=2
            h+=2
            notcnt+=1
            if notcnt % 7 == 0:
                x = int(Gww*0.1)
                y = int(Ghh*0.1)
                w = int(Gww*0.8)
                h = int(Ghh*0.8)
                case_not = 5
        
        # 임의로 박스 주변 crop 범위 지정
        x_ = int((2*x-w) / 2)
        y_ = int((2*y-h) / 2)
        w_ = int(2*w)
        h_ = int(2*h)
        xw = x_+w_
        yh = y_+h_

        if x_ <= 0: x_ = 0
        if y_ <= 0: y_ = 0
        if xw >= Gww: xw = Gww
        if yh >= Ghh: yh = Ghh
        
        print("box info :", x,y,x+w,y+h)
        
        # 바운딩 박스 좌표 값 저장
        X_min.append(x)
        X_max.append(x+w)
        Y_min.append(y)
        Y_max.append(y+h)
        
        ### 낙상 판단 방향
        case_d = 0
        if ((X_max[i] + Y_min[i])/2 > (X_min[i] + Y_max[i])/2) and i>0:
            #좌측낙사 case = 3
            if X_min[i] < X_min[i-1] and X_max[i] < X_max[i-1] and Y_min[i] > Y_min[i-1] and Y_max[i] > Y_max[i-1]:
                case_d = 3

            #우측낙사 case = 3
            elif X_min[i] > X_min[i-1] and X_max[i] > X_max[i-1] and Y_min[i] > Y_min[i-1] and Y_max[i] > Y_max[i-1]:
                case_d = 3         

            #전방낙사 case = 1
            elif Y_min[i] > Y_min[i-1] and Y_max[i] > Y_max[i-1]:
                case_d = 1

            #후방낙사 case = 2
            elif Y_min[i] > Y_min[i-1] :
                case_d = 2
        
        ### 속력 낙상 판단
        case_v = 0
        PX.append((X_max[i] + X_min[i])/2)
        PY.append((Y_max[i] + Y_min[i])/2)
        Pixel2cm = (Y_max[i-1]+Y_min[i-1])/170

        a = (X_max[i]+X_min[i])/2 - (X_max[i-1]+X_min[i-1])/2
        b = (Y_max[i]+Y_min[i])/2 - (Y_max[i-1]+Y_min[i-1])/2

        #속력 구하기
        V = (math.sqrt(a**2 + b**2) * Pixel2cm)/(1/2)

        if V >= 350 and V < 3000:
            if X_min[i-2] != int(Gww*0.1) and Y_min[i-2] != int(Ghh*0.1) and X_max[i-2] != int(Gww*0.9) and Y_max[i-2] != int(Ghh*0.9):
                case_v = 1
        elif V >= 300 and V < 350:
            case_v = 2
        else:
            case_v = 0
        i+=1
        
        # 정상일 때 자막 저장 & 이상 상황 발생 시 event 값 부여
        if event == 0:
            if case_not == 0 and case_d == 1 and case_v == 1:
                event = 1
            elif case_not == 0 and case_d == 2 and case_v == 1:
                event = 2
            elif case_not == 0 and case_d == 3 and case_v == 1:
                event = 3
            elif case_not == 0 and case_d > 0 and case_v == 2:
                event = 4
            else:
                print("<<<정상>>>")
                print("속도 :", V)
                qq = '정상'
        
        # 전방 낙상 경우
        if event == 1:
            print("<<<앞으로 넘어짐!>>>")
            print("속도 :", V)
            qq = '앞으로 넘어짐!'
            event_cnt+=1

        # 후방 낙상 경우
        if event == 2:
            print("<<<뒤로 넘어짐!>>>")
            print("속도 :", V)
            qq = '뒤로 넘어짐!'
            event_cnt+=1
                
        # 측면 낙상 경우
        if event == 3:
            print("<<<옆으로 넘어짐!>>>")
            print("속도 :", V)
            qq = '옆으로 넘어짐!'
            event_cnt+=1

        # 낙상이 의심되는 상황
        if event == 4:
            print("<<<낙상 의심>>>")
            print("속도 :", V)
            qq = '낙상 의심'
            event = 0
        
        # 10초 후 상태 초기화
        if event_cnt == 150:
            event = 0
            event_cnt = 0

        j+=1
        check+=1
        
        print("%d\n%02d:%02d:%02d,%03d --> %02d:%02d:%02d,%03d\n%s\n" % (j, hh1, mm1, ss1, ms1, hh2, mm2, ss2, ms2, qq), file=file_srt)
        
        for k in range(len(results.xyxy[0])):
            conf = results.pandas().xyxy[0].confidence[0]*100
            label = "person: {:.2f}".format(conf)
            color_g = (0, 255, 0)
            cv2.rectangle(image, (int(results.pandas().xyxy[0].xmin[0]), int(results.pandas().xyxy[0].ymin[0])), (int(results.pandas().xyxy[0].xmax[0]), int(results.pandas().xyxy[0].ymax[0])), color_g, 2)
            cv2.putText(image, label, (int(results.pandas().xyxy[0].xmin[0]), int(results.pandas().xyxy[0].ymin[0] - 5)), cv2.FONT_HERSHEY_DUPLEX, 1, color_g, 1)
        
        #결과 show
        cv2.imshow('Falling Detection', image)

        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q') or key_pressed == 27:
            break
        
        print(f"frame time: {(time() - tt0):.4f}, elapsed: {(time() - tt_i):.4f}----------")
                
file_srt.close()
        
cap.release()
cv2.destroyAllWindows()

