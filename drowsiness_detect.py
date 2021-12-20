import RPi.GPIO as GPIO
import numpy as np
from time import sleep
from scipy.spatial import distance as dist
import cv2
import dlib
import time

FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 460

modelPath = "/home/pi/project/shape_predictor_68_face_landmarks.dat"

thresh = 0.2
buzzer=25 #핀번호 22
led=17    #핀번호 17
drowsy = 0
drowsyLimit = 5

#GPIO.setup([buzzer, led1] ,GPIO.OUT) #GPIO핀을 out용도로 지정
GPIO.setwarnings(False)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

leftEyeIndex = [36, 37, 38, 39, 40, 41]
rightEyeIndex = [42, 43, 44, 45, 46, 47]

def bellAlert():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(buzzer, GPIO.OUT)

    GPIO.output(buzzer, GPIO.HIGH) #해당핀 출력
    GPIO.output(buzzer, True)
    sleep(3.0)
    GPIO.output(buzzer, GPIO.LOW)
    GPIO.output(buzzer, False)

def lampAlert():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(led, GPIO.OUT)

    GPIO.output(led, GPIO.HIGH)
    GPIO.output(led, True)
    sleep(3.0)
    GPIO.output(led, GPIO.LOW)
    GPIO.output(led, False)

def eye_aspect_ratio(eye): # EAR 계산 함수
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear

def grayscaling(image): # 그레이스케일
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # image 파일을 BGR에서 단일색상 gray로 변경
    return cv2.equalizeHist(gray) # 히스토그램 평활화 ( 명암비 향상 기법 )

def checkDrowsy(eyeStatus, t_list):
    drowsy = 0
    if eyeStatus:       #눈 뜬상태
        t_list = []      #시간 초기화

    else:   #눈을 감았을 때만..
        t_list.append(time.time())     #현재시간을 추가..

        #drowsyLimit : 임의로 정해야하는 졸음 판별 임계값
        #차가 달리는 속도에 따라서 임계값을 바꿔야할지는 고민이 필요할 것 같아요..
        if t_list[-1] - t_list[0] > drowsyLimit:
            drowsy = 1
    return drowsy   #졸음 : 1,  아님 : 0

def checkEyeStatus(landmarks, frame):
    mask = np.zeros(frame.shape[:2], dtype = np.float32)

    #양쪽 눈의 좌표를 가져와서 모양을 따온다
    #fillConvexPoly : 주어진 점으로 이루어진 볼록다각형 만들어준다
    hullLeftEye = []
    for i in range(0, len(leftEyeIndex)):
        hullLeftEye.append((landmarks[leftEyeIndex[i]][0], landmarks[leftEyeIndex[i]][1]))
    cv2.fillConvexPoly(mask, np.int32(hullLeftEye), 255)

    hullRightEye = []
    for i in range(0, len(rightEyeIndex)):
        hullRightEye.append((landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]))
    cv2.fillConvexPoly(mask, np.int32(hullRightEye), 255)

    #EAR값 구하기..
    leftEAR = eye_aspect_ratio(hullLeftEye)
    rightEAR = eye_aspect_ratio(hullRightEye)

    ear = (leftEAR + rightEAR) / 2.0

    eyeStatus = 1          # 1 -> Open, 0 -> closed

    if (ear < thresh):
        eyeStatus = 0

    return eyeStatus

def facial_landmark(image):
    imSmall = cv2.resize(image, None,
                            fx = 1.0/FACE_DOWNSAMPLE_RATIO,
                            fy = 1.0/FACE_DOWNSAMPLE_RATIO,
                            interpolation = cv2.INTER_LINEAR)

    rects = detector(imSmall, 0)
    if len(rects) == 0:
        return 0

    newRect = dlib.rectangle(int(rects[0].left() * FACE_DOWNSAMPLE_RATIO),
                            int(rects[0].top() * FACE_DOWNSAMPLE_RATIO),
                            int(rects[0].right() * FACE_DOWNSAMPLE_RATIO),
                            int(rects[0].bottom() * FACE_DOWNSAMPLE_RATIO))

    points = []
    [points.append((p.x, p.y)) for p in predictor(image, newRect).parts()]
    return points

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    cv2.namedWindow('Drowsiness Detect')

    while True:
        try:
            ret, frame = capture.read()
            height, width = frame.shape[:2]
            IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
            frame = cv2.resize(frame, None,
                                fx = 1/IMAGE_RESIZE,
                                fy = 1/IMAGE_RESIZE,
                                interpolation = cv2.INTER_LINEAR)

            frame_gray = grayscaling(frame)
            landmarks = facial_landmark(frame_gray)

            if landmarks == 0:
                cv2.putText(frame, "Unable to detect face, Please check proper lighting", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "or decrease FACE_DOWNSAMPLE_RATIO", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow("Drowsiness Detect", frame)
                time_checker = []
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            else:
                cv2.putText(frame, "Face detecting...", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow("Drowsiness Detect", frame)

            for i in range(0, len(leftEyeIndex)):
                cv2.circle(frame, (landmarks[leftEyeIndex[i]][0], landmarks[leftEyeIndex[i]][1]), 1, (0, 0, 255), -1, cv2.LINE_AA, 0)

            for i in range(0, len(rightEyeIndex)):
                cv2.circle(frame, (landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]), 1, (0, 0, 255), -1, cv2.LINE_AA, 0)

            eye_status = checkEyeStatus(landmarks, frame)
            drowsy = checkDrowsy(eye_status, time_checker)

            if drowsy:
                cv2.putText(frame, "! ! ! ! DROWSINESS ALERT ! ! ! !", (70, 50), 3, 1, (0, 0, 255), 2, cv2.LINE_AA)
                bellAlert()
                lampAlert()
                time_checker = []
                drowsy = 0
            else:
                cv2.putText(frame, "Please wear your seat belt at all times", (70, 50), 3, 0.8, (255,0,0), 2, cv2.LINE_AA)

            cv2.imshow("Drowsiness Detect", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        except Exception as e:
            print("Error Occured\n")
            print(e)

    GPIO.cleanup()
    capture.release()
    cv2.destroyAllWindows()