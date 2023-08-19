import cv2
import mediapipe as mp
import PoseModel as pm
import time

cap = cv2.VideoCapture(0)
cTime = 0
pTime = 0
detector = pm.poseEstimation()

while True:
    _, frame = cap.read()

    frame = detector.findPose(frame)
    lmList = detector.getPose(frame, draw=False)

    print(lmList[14])
    cv2.circle(frame, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
