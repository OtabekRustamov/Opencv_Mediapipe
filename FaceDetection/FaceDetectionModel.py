import cv2
import mediapipe as mp
import time


class FaceDetector:

    def __int__(self, minDetectionConf=0.5,model_selection = 0):
        faceDetection = 0
        self.minDetectionConf = minDetectionConf
        self.model_selection = model_selection

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConf,
                                                                self.model_selection)

    def findFace(self, frame, draw=True):

        bboxs = []

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.faceDetection.process(frameRGB)

        if result.detections:
            for id, detection in enumerate(result.detections):
                # mpDraw.draw_detection(frame,detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                fh, fw, fc = frame.shape
                bbox = int(bboxC.xmin * fw), int(bboxC.ymin * fh), \
                    int(bboxC.width * fw), int(bboxC.height * fh)
                bboxs.append([id, bbox, detection.score])

                cv2.rectangle(frame, bbox, (255, 0, 255), 2)
                cv2.putText(frame, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            3, (255, 0, 0))
        return frame, bboxs


def main():
    cap = cv2.VideoCapture(0)
    cTime = 0
    pTime = 0
    detection = FaceDetector()
    while True:

        _, frame = cap.read()
        frame, bboxs = detection.findFace(frame)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0))
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
