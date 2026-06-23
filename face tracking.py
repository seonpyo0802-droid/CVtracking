import mediapipe as mp
import cv2
import cvzone as cz
import time

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

pTime = 0
run = True

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()
mpDraw = mp.solutions.drawing_utils

while run:
    success, img = cap.read()
    results = faceDetection.process(img)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    if results.detections:
        for id,detection in enumerate(results.detections):
            mpDraw.draw_detection(img,detection)

    cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv2.imshow("Setup", img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        run = False