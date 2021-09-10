# !pip install mmcv-full
# !git clone https://github.com/open-mmlab/mmdetection.git
# !cd mmdetection; python setup.py install

import cv2
import time
import numpy as np
import mmcv
from mmdet.apis import init_detector, inference_detector

# !cd mmdetection; mkdir checkpoints
# !wget -O /content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# !ls -lia /content/mmdetection/checkpoints

config_file = '/content/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/content/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

model = init_detector(config_file, checkpoint_file)
# !mkdir data
# !ls -lia /content/data

# by 종두 model로 image detection test@@
# import cv2
# from mmdet.apis import show_result_pyplot

# img_arr = cv2.imread('mmdetection/demo/demo.jpg')
# results = inference_detector(model, img_arr)

# show_result_pyplot(model, img_arr, results)



# 0부터 순차적으로 클래스 매핑된 label 적용. 
labels_to_names_seq = {0:'person'}

def get_detected_image(model, img_array, score_threshold, is_print=True):
    draw_img = img_array.copy()
    box_color = (0, 255 ,0)
    text_color = (0, 0, 255)

    results = inference_detector(model, draw_img)

    for result_idx, result in enumerate(results):
        if len(result) == 0:
            continue
        if result_idx > 0:
            break

        result_filtered = result[np.where(result[:, 4] > score_threshold)]

        for i in range(len(result_filtered)):
            left = int(result_filtered[i, 0])
            top = int(result_filtered[i, 1])
            right = int(result_filtered[i, 2])
            bottom = int(result_filtered[i, 3])
            score = result_filtered[i, 4]

            caption = '{}:{:.2f}'.format(labels_to_names_seq[result_idx], score)

            cv2.rectangle(draw_img, (left, top), (right, bottom), color=box_color, thickness=2)
            cv2.putText(draw_img, caption, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

            if is_print:
                print(caption)
    return draw_img

def do_detected_video(model, input_path, output_path, threshold ,is_print=True):
    cap = cv2.VideoCapture(input_path)
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_writer = cv2.VideoWriter(output_path, codec, video_fps, video_size)

    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('총 frame 개수:{}' .format(frame_cnt))

    start = time.time()
    while True:
        hasFrame, img_frame = cap.read()

        if not hasFrame:
            print('더 이상 Frame이 없습니다.')
            break
        
        img_frame = get_detected_image(model, img_frame, score_threshold=threshold, is_print=True)
        video_writer.write(img_frame)

    video_writer.release()
    cap.release()

    print('총 걸린 시간 : {:.2f}'.format(time.time() - start))

do_detected_video(model, 'vtest.avi', 'video_out_mmdetection.avi', threshold=0.5, is_print=True)