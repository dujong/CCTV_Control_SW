# CCTV_Control_SW
### Development of Intelligent CCTV Control SoftWare to Minimize Blind Spot  
+ 일반 CCTV와 특수 CCTV 조합으로 관제 시스템을 구성하고 일반 CCTV와 특수 CCTV를 통합 관리할 수 있는 지능형 관제 소프트웨어를 개발  
+ 사각지대 최소화를 위한 특수 CCTV는 일반 CCTV에 비해 낮은 해상도지만 넓은 지역을 관제하고, 저비용의 장점을 갖도록 설계  
+ 추적 객체가 일반 CCTV의 사각지대 진입 시, 특수 CCTV로 촬영하여 객체의 이동 추적, 특수 CCTV는 객체가 특정 위치에 있는지만 확인하고 추적 객체가 다른 일반 CCTV 구역으로 이동할 때까지만 위치를 추적  
+ 특수 CCTV는 객체가 특정 위치에 있는지만 확인하고 추적 객체가 다른 일반 CCTV 구역으로 이동할 때까지만 위치를 추적  <br><br>

## A Sector
![A](https://user-images.githubusercontent.com/55770741/139579449-e3874d92-c72e-4c9b-a115-0a5b37233885.PNG)<br>

## Critical Sector
![cz-2](https://user-images.githubusercontent.com/55770741/139579415-7e6c0d68-7907-48da-9827-948caaadef4e.PNG)<br>

## B Sector
![B](https://user-images.githubusercontent.com/55770741/139579370-e3f42b34-a548-4697-b179-3d5667400485.PNG)<br><br>  

# Review
1. Tracker라는 Library를 처음 활용해봤는데, 이미 구현되어 있는 Library를 목적에 맞게 활용하기 위해서 내부 소스코드를 이해 및 수정하면서, 내부 구현을 손보는 일에 대한 두려움이 사라지고, 내가 원하는 대로 바꾸어 사용하니 재미를 느꼇다.
2. 사람에 대한 특징 상의, 하의를 Color 추출할 때 Detector한 Bounding Box를 활용해서 추출하였는데, 배경색이 같이 추출되어져 Issue가 있었다. 이러한 문제를 해결하기 위해서 Segmentation 기술을 사용하여 배경을 제외한 상의와 하의만을 인식하여 Color 추출을 하면 더욱 성능이 향상될 것이라고 생각한다.
3. 현재 사용자의 특징을 추출하는 요소들이 상의, 하의에 국한되어 있지만, 추후에 사용자의 얼굴을 고려하여 판별해주는 CCTV Control SW를 개발하고 싶다는 생각이 들었다

# Requirement  
+ Python
+ Opencv
+ Tensorflow
+ Pandas
+ Numpy
+ Tracker library
+ yolov4 weight model

# 참조한 곳
https://github.com/theAIGuysCode/yolov4-deepsort
