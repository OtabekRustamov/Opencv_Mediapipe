import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
mp_drawing_styles = mp.solutions.drawing_styles

cTime = 0
pTime = 0

while True:

    _, frame = cap.read()
    fraameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = faceMesh.process(frame)
    if result.multi_face_landmarks:
        for faceLM in result.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faceLM, mpFaceMesh.FACEMESH_TESSELATION,
                                  drawSpec, drawSpec)
            # mpDraw.draw_landmarks(frame,faceLM, mpFaceMesh.FACEMESH_TESSELATION,                                      landmark_drawing_spec=None,
            #   connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            for id, lm in enumerate(faceLM.landmark):
                fh, fw, fc = frame.shape
                x,y = int(lm.x*fh),int(lm.y*fw)

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
