import mediapipe as mp
import cv2
import cvzone as cz
import time

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

pTime = 0
run = True

mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(static_image_mode = False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mpDraw=mp.solutions.drawing_utils
drawingSpec=mpDraw.DrawingSpec(color=(255,255,255), thickness=1,circle_radius=2)

while run:
    success, img= cap.read()
    #img=cv2.imread("CD/image/obama1.jpg")
    results=faceMesh.process(img)
    
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawingSpec,drawingSpec)
            for id,lm in enumerate(faceLms.landmark):
                ih,iw,ic=img.shape
                x,y=int(lm.x*iw),int(lm.y*ih)
                print(f'ID:{id} , X:{x} , Y:{y}')
    
    cTime = time.time() #current time
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(204,102,0),3)
    cv2.putText(img,'Ryan, 2023, July 6th, SetUp',(800,700),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3) #(where,text,coordinate,font,font size,color,spacing)
    cv2.imshow('Setup', img)
    key = cv2.waitKey(1) #slowing down the game
    
    if key == ord('q'):
        run = False
    
    if key == ord('1'):
        img=cv2.imread("CD/image/obama1.jpg")