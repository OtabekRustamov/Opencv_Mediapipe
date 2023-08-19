import cv2
import mediapipe as mp
import time


def fanyDraw(frame, bbox, l=20, t=8):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

    cv2.rectangle(frame, bbox, (250, 0, 255), 2)
    # Top Left x y
    cv2.line(frame, (x, y), (x + l, y), (255, 0, 255), t)
    cv2.line(frame, (x, y), (x, y+l), (255, 0, 255), t)

    # Top Right x1 y
    cv2.line(frame, (x1, y), (x1 - l, y), (255, 0, 255), t)
    cv2.line(frame, (x1, y), (x1, y + l), (255, 0, 255), t)

    # Buttom Left x y1
    cv2.line(frame, (x, y1), (x + l, y1), (255, 0, 255), t)
    cv2.line(frame, (x, y1), (x, y1 - l), (255, 0, 255), t)

    # Buttom Left x1 y1
    cv2.line(frame, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
    cv2.line(frame, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

    return frame


cap = cv2.VideoCapture(0)
cTime = 0
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:

    _, frame = cap.read()

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = faceDetection.process(frameRGB)

    if result.detections:
        for id, detection in enumerate(result.detections):
            # mpDraw.draw_detection(frame,detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            fh, fw, fc = frame.shape
            bbox = int(bboxC.xmin * fw), int(bboxC.ymin * fh), \
                int(bboxC.width * fw), int(bboxC.height * fh)

            # cv2.rectangle(frame, bbox, (255, 0, 255), 2)
            frame = fanyDraw(frame, bbox)

            cv2.putText(frame, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        3, (255, 0, 255))

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                5, (255, 0, 0))
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
