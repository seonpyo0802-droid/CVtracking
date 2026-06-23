import mediapipe as mp
import cv2
import cvzone as cz
import time

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

mpHandDetection = mp.solutions.hands
handDetection = mpHandDetection.Hands(max_num_hands=2, min_detection_confidence = 0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
run = True

while run:
    success, img= cap.read()
    results = handDetection.process(img)
    
    cTime = time.time() #current time
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                print(id,lm)#id = number of landmark, lm = location
                h,w,c = img.shape#getting the height, width, and center of the screen
                cx,cy = int(lm.x*w),int(lm.y*h)#by the x and y coordinate of lm, which is the location of the landmarks, we multiply them to the width and the height of the screen to find the landmark's real coordinates

                if id==4:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
                elif id==8:
                    cv2.circle(img, (cx,cy), 5, (0,0,255), cv2.FILLED)
                elif id==12:
                    cv2.circle(img, (cx,cy), 5, (0,255,0), cv2.FILLED)
                elif id==16:
                    cv2.circle(img, (cx,cy), 5, (0,0,0), cv2.FILLED)
                elif id==20:
                    cv2.circle(img, (cx,cy), 5, (0,150,130), cv2.FILLED)
                else:
                    cv2.circle(img, (cx,cy), 5, (255,255,255), cv2.FILLED)
    
    cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(204,102,0),3)
    cv2.putText(img,'Ryan, 2023, July 6th, SetUp',(800,700),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3) #(where,text,coordinate,font,font size,color,spacing)
    cv2.imshow('Setup', img)
    key = cv2.waitKey(1) #slowing down the game
    
    if key == ord('q'):
        run = False
    
    