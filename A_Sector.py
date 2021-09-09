import os
import cv2
import random
input_path = 'vtest.avi'

cap = cv2.VideoCapture('vtest.avi')
greeb_box = (0, 255, 0)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
i = 0
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
        
    detected, _ = hog.detectMultiScale(frame)
    
    # 검출 결과 화면 표시
    for data in detected:
        xmin = data[0]
        ymin = data[1]
        xmax = xmin + data[2]
        ymax = ymin + data[3]
        print('cnt:{} x:{}, y:{}, w:{}, h:{}'.format(i, xmin, ymin, xmax, ymax))
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), greeb_box, thickness=3)
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(10) == 27:
        break
    i += 1

cv2.destroyAllWindows()