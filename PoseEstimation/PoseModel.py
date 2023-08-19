import cv2
import mediapipe as mp
import numpy as np
import time
import math


class poseEstimation:
    def __init__(self, mode=False, modComplexity=1, smooth=True,
                 eSegmentation=False, sSegmentation=True, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.modComplexity = modComplexity
        self.smooth = smooth
        self.eSegmentation = eSegmentation
        self.sSegmentation = sSegmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modComplexity,
                                     self.smooth, self.eSegmentation, self.sSegmentation,
                                     self.detectionCon, self.trackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findPose(self, frame, draw=True):

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(imgRGB)
        if draw:
            if self.result.pose_landmarks:
                self.mpDraw.draw_landmarks(frame, self.result.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)

        return frame

    def getPose(self, frame, draw=True):

        self.lmPose = []
        if self.result.pose_landmarks:
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmPose.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmPose

    def findAngle(self, frame, p1, p2, p3, draw=True):

        # Land maks
        x1, y1 = self.lmPose[p1][1:]
        x2, y2 = self.lmPose[p2][1:]
        x3, y3 = self.lmPose[p3][1:]

        # Angle clacuation
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x2 - x1))
        if angle < 0:
            angle += 360
        # print(angle)
        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
            cv2.line(frame, (x3, y3), (x2, y2), (0, 0, 0), 3)
            cv2.circle(frame, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x1, y1), 5, (255, 0, 255), 2)
            cv2.circle(frame, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 5, (255, 0, 255), 2)
            cv2.circle(frame, (x3, y3), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x1, y1), 5, (255, 0, 255), 2)
            cv2.putText(frame,str(int(angle)),(x2 - 50,y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
        return angle


def main():
    cap = cv2.VideoCapture(0)
    cTime = 0
    pTime = 0
    detector = poseEstimation()

    while True:
        _, frame = cap.read()

        frame = detector.findPose(frame)
        lmList = detector.getPose(frame)

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


if __name__ == "__main__":
    main()

# import mediapipe as mp
# import time
# class PoseDetector:
#
#     def __init__(self, mode = False, upBody = False, smooth=True, detectionCon = 0.5, trackCon = 0.5):
#
#         self.mode = mode
#         self.upBody = upBody
#         self.smooth = smooth
#         self.detectionCon = detectionCon
#         self.trackCon = trackCon
#
#         self.mpDraw = mp.solutions.drawing_utils
#         self.mpPose = mp.solutions.pose
#         self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
#                                      self.detectionCon, self.trackCon)
#
#     def findPose(self, img, draw=True):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.pose.process(imgRGB)
#         #print(results.pose_landmarks)
#         if self.results.pose_landmarks:
#             if draw:
#                 self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
#
#         return img
#
#     def getPosition(self, img, draw=True):
#         lmList= []
#         if self.results.pose_landmarks:
#             for id, lm in enumerate(self.results.pose_landmarks.landmark):
#                 h, w, c = img.shape
#                 #print(id, lm)
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 lmList.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
#         return lmList
#
# def main():
#     cap = cv2.VideoCapture('videos/a.mp4') #make VideoCapture(0) for webcam
#     pTime = 0
#     detector = PoseDetector()
#     while True:
#         success, img = cap.read()
#         img = detector.findPose(img)
#         lmList = detector.getPosition(img)
#         print(lmList)
#
#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime
#
#         cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
#         cv2.imshow("Image", img)
#         cv2.waitKey(1)
#
#
# if __name__ == "__main__":
#   main()
