# Software-Convergence-Project

1. 과제 개요 : 
 자동차는 현대사회에 이르러 인간의 삶과 매우 밀접하게 연관되어 있다. 삶의 질을 향상시키고 이동 시간을 줄여주지만 자동차가 만들어낸 심각한 문제점이 있다. 바로 졸음운전으로 인한 교통사고이다. 
이러한 문제점을 해결하기 위해 졸음운전을 방지할 수 있는 시스템을 만들어 보기로 하였다. 우선 라즈베리파이에 파이 카메라와 LED경고등을 연결한다. 그 다음 카메라를 통해 운전자의 눈 움직임을 실시간으로 촬영해 이미지로 처리하고 영상에서 운전자의 눈 깜박임 빈도수를 감지하여 졸음 운전으로 인식하도록 설정한 빈도수에 가까워 지면 경고등이 켜진다. 마지막으로 일정한 수치를 넘으면 피에조 부저를 이용해 경고음을 작동시킨다. 이미 경고음을 활용한 졸음운전방지시스템이 나와 있어서 우리팀은 좀 더 발전시켜 경고등과 경고음 둘다 활용한 시스템을 계획하였다. 이 시스템이 기존 시스템보다 졸음운전으로 한 사고를 더 줄여줄 것이라 확신한다.

2. 과제 수행 내용 : 
1) 라즈베리 파이에 웹캠 연결
- 라즈베리 파이와 웹캠을 연결해 눈 깜박임을 감지하여 데이터를 전송한다.
2) 라즈베리파이에 비상등과 경고음을 위한 센서 연결
- 라즈베리파이에 비상등과 경고음을 위해 피에조부저, LED 경고 등 센서를 연결한다.
3) 라즈베리 파이에 OS 설치 후 프로그래밍
- 라즈베리 파이에서 웹캠을 통해 촬영한 영상을 분석할 수 있도록 OpenCV 라이브러리를 활용해 프
로그래밍을 한다.
4) 웹캠으로부터 받은 데이터를 라즈베리파이로 전송 후 센서 발동
- 라즈베리파이에서 센서가 작동할 수 있도록 영상분석한 데이터를 전송한다. 데이터가 설정한 값보
다 수치가 작으면 졸음운전으로 판단 후 연결한 센서를 발동해 졸음운전을 방지한다.
5) 완성된 작품 꾸미기
- 라즈베리 파이와 각종 센서들을 다 연결 후 제대로 작동을 한다면 외형을 경찰 버스 모형으로 만들
어 실제로 사용하는 사람들이 더 편하게 사용할 수 있도록 한다.

3. 과제 목표 :
 프로젝트 결과물은 경찰차 형태로 제작할 것이다. 7인치 스크린을 통해 운전자의 영상을 실시간으로 
내보내며, 해당 스크린은 경찰차의 창문에 부착한다. 스크린을 통해 영상을 내보내는 의도는 동승자에
게 운전자의 상태를 쉽게 파악할 수 있도록 하기 위함이다. 경고등은 경찰차의 경광등으로 제작하며, 
경고음은 경찰차의 사이렌 소리처럼 표현될 것이다. 디자인은 시각적으로 호불호가 갈리지 않는 형태
로 구상한다.
실제로 졸음운전을 하지 않았는데 졸음운전으로 판단하여 경고하는 횟수를 최소화하고, 졸음운전을 
하였을 때 졸음운전으로 판단하지 못하는 경우가 없도록 주의한다. 이러한 과정을 거쳐 시스템의 정확
도가 90% 이상을 달성하는 것이 목표이다
