import os
import cv2
import random
input_path = 'vtest.avi'

cap = cv2.VideoCapture('vtest.avi')
greeb_box = (0, 255, 0)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vid_fps = cap.get(cv2.CAP_PROP_FPS)
vid_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
output_path = cv2.VideoWriter('vtest_out.avi', codec, vid_fps, vid_size)

i = 0
while True:
    hasFrame, img_frame = cap.read()
    
    if not hasFrame:
        break
    
    detected, test = hog.detectMultiScale(img_frame)
    
    # 검출 결과 화면 표시
    for data in detected:
        xmin = data[0]
        ymin = data[1]
        xmax = xmin + data[2]
        ymax = ymin + data[3]
        print('cnt:{} x:{}, y:{}, w:{}, h:{}'.format(i, xmin, ymin, xmax, ymax))
        cv2.rectangle(img_frame, (xmin, ymin), (xmax, ymax), greeb_box, thickness=3)
        
    output_path.write(img_frame)
    cv2.imshow('result', img_frame)
    if cv2.waitKey(1) == ord('q'):
        break
    i += 1

cap.release()
output_path.release()