import cv2
import mediapipe as mp
import numpy as np
import time
from PoseEstimation import PoseModel as pm

cTime = 0
pTime = 0
cap = cv2.VideoCapture(0)
detector = pm.poseEstimation()
count = 0
dir = 0

while True:

    _, frame = cap.read()

    # frame = cv2.imread("train.jpg")
    frame = detector.findPose(frame, False)
    lmList = detector.getPose(frame, False)
    if len(lmList) != 0:
        # # Right hand
        # detector.findAngle(frame,12,14,16)
        #     Left hand
        #     detector.findAngle(frame, 11, 13, 15)

        angle = detector.findAngle(frame, 11, 13, 15)
        per = np.interp(angle,(210,310),(0,100))

        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1

        if per == 100:
            if dir == 1:
                count += 0.5
                dir = 0

        cv2.putText(frame, str(int(count)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0), 5)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
    #             3, (255, 0, 0), 3)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
