import time
import cv2
from HandDetection import HandTrackingModel as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

wCam, hCam = 480, 480
pTime = 0
cTime = 0
vol = 0
volBar = 400
volPer = 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange()

minVolume = volumeRange[0]
maxVolume = volumeRange[1]

cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

cap.set(3, wCam)
cap.set(4, hCam)

while True:
    _, frame = cap.read()

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList) != 0:
        # print(lmList[4])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        distance = math.hypot(x2-x1,y2-y1)
        # print(distance)
    #     Hand range 50 - 300
    #     Volume range -65 - 0
        vol = np.interp(distance,[50,250],[minVolume,maxVolume])
        volBar = np.interp(distance, [50, 250], [400, 150])
        volPer = np.interp(distance, [50, 250], [0, 100])
        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)
        if distance < 30:
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(frame,(50,150),(85,400),(255, 0, 0),3)
    cv2.rectangle(frame, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(frame, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 255), 3)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
