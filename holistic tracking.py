import mediapipe as mp
import cv2
import cvzone as cz
import time

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

pTime = 0
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
mpDrawStyle = mp.solutions.drawing_styles

run = True

while run:
    success, img= cap.read()
    results=pose.process(img)
    
    cTime = time.time() #current time
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            ih,iw,ic=img.shape
            x,y=int(lm.x*iw),int(lm.y*ih)
            if id<=10:
                cv2.circle(img,(x,y), 3,(0,0,255), cv2.FILLED)
            elif 10<id<=22:
                cv2.circle(img,(x,y), 3,(0,255,0), cv2.FILLED)
            elif 22<id<=24:
                cv2.circle(img,(x,y), 3,(0,255,255), cv2.FILLED)
            elif 24<id<=32:
                cv2.circle(img,(x,y), 3,(255,0,0), cv2.FILLED)
    
    cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(204,102,0),3)
    cv2.putText(img,'Ryan, 2023, July 6th, SetUp',(800,700),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3) #(where,text,coordinate,font,font size,color,spacing)
    cv2.imshow('Setup', img)
    key = cv2.waitKey(1) #slowing down the game
    
    if key == ord('q'):
        run = False
    
    