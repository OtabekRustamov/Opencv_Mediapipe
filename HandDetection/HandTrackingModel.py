import time
import mediapipe as mp
import cv2


class HandDetector():

    def __init__(self, model=False, maxHands=1, detectionConf=0.5, trackConf=0.5):
        self.model = model
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=3)

    def findHands(self, frame, draw=True):

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handLm in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLm, self.mpHands.HAND_CONNECTIONS)
                                               # self.drawSpec,self.drawSpec)

        return frame
    def findPosition(self,img,handNo = 0,draw = True):

        lmList =[]
        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                #                 print(id,lm)
                h, w, ch = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
        return lmList


# def main():
#     pTime = 0
#     cTime = 0
#     cap = cv2.VideoCapture(0)
#     detector = handDetector()
#
#     while True:
#         _, frame = cap.read()
#
#         frame = detector.findHands(frame)
#         lmList = detector.findPosition(frame)
#         if len(lmList) != 0:
#             print(lmList[4])
#
#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime
#
#         cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#         cv2.imshow("frame", frame)
#         if cv2.waitKey(1) == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()
