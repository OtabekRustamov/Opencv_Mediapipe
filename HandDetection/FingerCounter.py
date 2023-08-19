import os
import time
import cv2
import HandTrackingModel as htm
import math
import matplotlib.pyplot as plt

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)

detector = htm.HandDetector()
folderPath = "C://Users//HP//PycharmProjects//pythonProject//FingerData//"
myList = os.listdir(folderPath)
# print(myList)
imgList = []
tipIds = [4, 8, 12, 16, 20]

for imPath in myList:
    img = cv2.imread(f'{folderPath}/{imPath}')
    print(f'{folderPath}/{imPath}')
    imgList.append(img)
# print(len(imgList))

while True:
    _, frame = cap.read()

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)

    if len(lmList) != 0:
        fingers = []
        # thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                # print('Index finger is open')
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFing = fingers.count(1)
        print(totalFing)

        h, w, ch = imgList[totalFing - 1].shape
        frame[0:h, 0:w] = imgList[totalFing - 1]

        cv2.rectangle(frame, (20, 255), (170, 445), (255, 0, 0), cv2.FILLED)
        cv2.putText(frame, (str(totalFing)), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 255), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
