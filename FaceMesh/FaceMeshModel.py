import cv2
import mediapipe as mp
import time


class FaceMeshDetector():

    def __init__(self, static_image_mode=False, max_num_faces=2,refine_landmarks=False,
                 min_detection_confidence=0.5,min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.static_image_mode = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode,self.static_image_mode,
                                                 self.refine_landmarks,self.min_detection_confidence,
                                                 self.min_tracking_confidence)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def findFaceMesh(self,frame,draw = True):

        faces = []
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.faceMesh.process(frameRGB)

        if result.multi_face_landmarks:
            for faceLM in result.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, faceLM, self.mpFaceMesh.FACEMESH_TESSELATION,
                                      self.drawSpec, self.drawSpec)
                # mpDraw.draw_landmarks(frame,faceLM, mpFaceMesh.FACEMESH_TESSELATION,                                      landmark_drawing_spec=None,
                #   connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                face = []
                for id, lm in enumerate(faceLM.landmark):
                    fh, fw, fc = frame.shape
                    x, y = int(lm.x * fh), int(lm.y * fw)
                    cv2.putText(frame,str(id), (x,y), cv2.FONT_HERSHEY_PLAIN,
                                0.5, (255, 0, 255), 1)
                    face.append([x,y])
            faces.append(face)
        return frame
def main():
    cap = cv2.VideoCapture(0)

    cTime = 0
    pTime = 0

    detector = FaceMeshDetector()
    while True:

        _, frame = cap.read()
        frame = detector.findFaceMesh(frame)

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

if __name__ == "__main__":
    main()