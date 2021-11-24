def eye_aspect_ratio(eye): # EAR 계산 함수
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear

def histogram_equalization(image): # 그레이스케일
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # image 파일을 BGR에서 단일색상 gray로 변경
    return cv2.equalizeHist(gray) # 히스토그램 평활화 ( 명암비 향상 기법 )

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        try:
            ret, frame = capture.read()

            frame_gray = histogram_equalization(frame)
            landmarks = getLandmarks(frame_gray)

            for i in range(0, len(leftEyeIndex)):
                cv2.circle(frame, (landmarks[leftEyeIndex[i]][0], landmarks[leftEyeIndex[i]][1]), 1, (0, 0, 255), -1, cv2.LINE_AA, 0)

            for i in range(0, len(rightEyeIndex)):
                cv2.circle(frame, (landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]), 1, (0, 0, 255), -1, cv2.LINE_AA, 0) 

            eye_status = checkEyeStatus(landmarks)
            checkDrowsy(eye_status)

            if drowsy:
                cv2.putText(frame, "! ! ! ! DROWSINESS ALERT ! ! ! !", (70, 50), 3, 1, (0, 0, 255), 2, cv2.LINE_AA)
                soundAlert()
                lightAlert()
            else:
                cv2.putText(frame, "Please wear your seat belt at all times", (460, 80), 3, 0.8, (0,0,255), 2, cv2.LINE_AA)

            cv2.imshow("Drowsiness Detect", frame)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        except Exception as e:
            print("Error Occured\n")
            print(e)

    capture.release()
    cv2.destroyAllWindows()

        




