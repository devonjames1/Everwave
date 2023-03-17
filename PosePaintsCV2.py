import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import poseModule as pm
import math
import numpy as np
import tkinter as tk
import customtkinter
from tkinter import Variable, StringVar, IntVar, DoubleVar, BooleanVar
from tkinter.constants import *
import tkinter.filedialog as filedialog
CANV_WIDTH = 1920
CANV_HEIGHT = 1280


customtkinter.set_appearance_mode('dark')
customtkinter.set_default_color_theme('dark-blue')

root = customtkinter.CTk()
root.geometry("500x350")
ProgramPick = 0


def Pygui():
            
    frame = customtkinter.CTkFrame(master=root)
    frame.pack(pady=20, padx=60, fill='both', expand = True)

    label = customtkinter.CTkLabel(master=frame, text='Login System')
    label.pack(pady=12, padx=10)

    # entry1=customtkinter.CTkEntry(master=frame, placeholder_text='Username')
    # entry1.pack(pady=12, padx=10)

    # entry2=customtkinter.CTkEntry(master=frame, placeholder_text='Password')
    # entry2.pack(pady=12, padx=10)

    button1 = customtkinter.CTkButton(master=frame, text='Hand Det.', command = handDet)
    button1.pack(pady=12, padx=10)
    button2 = customtkinter.CTkButton(master=frame, text='Pose Det.', command = poseDet)
    button2.pack(pady=12, padx=10)
    checkbox = customtkinter.CTkCheckBox(master=frame, text='Welcome back?')
    checkbox.pack(pady=12, padx=10)
    

    # Display the code
    root.mainloop()
    print('Launcher Closed:', time.ctime())



pTime = 0
cTime = 0
# Initialize webcam #
# Insert 0 for integrated webcam
# Insert 1 for USB webcam
cap = cv2.VideoCapture(0)
# Set up moving average filter
num_frames = 60 # this is the num of frames back it will go to smooth.
                # higher number, smoother, but tiny more delay. It's actually not bad.
prev_landmarks = []
for i in range(num_frames):
    prev_landmarks.append(None)


# This is calling HandTrackingModule's handDetector() class
detector = htm.handDetector()
poseDetector = pm.poseDetector()


# Option 1 - Pose Detector
def PoseDetection():

    pTime = 0
    cTime = 0
    # Initialize webcam #
    # Insert 0 for integrated webcam
    # Insert 1 for USB webcam
    cap = cv2.VideoCapture(0)
    imgCanvas = np.zeros((CANV_HEIGHT, CANV_WIDTH, 3), np.uint8)# defining canvas

    while True:
        success, img = cap.read()
        xp, yp = 0, 0
        # Start the hand detector, and run it based off img
        img = poseDetector.findPose(img, draw=True)
        # Find the x,y coordinates of ALL hand landmarks
        lmList = poseDetector.findPosition(img, draw=False)
        
        img = cv2.resize(img, [CANV_WIDTH,CANV_HEIGHT])
    

        # CONSOLE PRINTING OPTIONS #
        if len(lmList) != 0:
            # break a leg!        
            leftIndex_X, leftIndex_Y = lmList[19][1], lmList[19][2]
            rightIndex_X, rightIndex_Y = lmList[20][1], lmList[20][2]
            #######################################
            cv2.circle(img, (leftIndex_X, leftIndex_Y), 30, (255,100,255), -1)
            
            # Apply moving average filter to smooth out landmark movement
            # Jesus christ holy fuck that was difficult
            # This is so cool
            def MovingAverage():
                prev_landmarks = 50
                filtered_landmarks = []
                for i in range(len(lmList)):
                    if prev_landmarks:
                        filtered_landmarks.append(lmList)
                    else:

                        Filter_ = ((num_frames - 1) * prev_landmarks[i] + lmList[i]) / num_frames
                        filtered_landmark = Filter_
                        filtered_landmarks.append(filtered_landmark)

                # Draw lines between the hand landmarks to create a drawing effect
                for i in range(len(filtered_landmarks)  ):
                    filtered_landmarks[i]
                    if i >= 0:
                            
                        devon= filtered_landmarks[i] * img.shape[1]
                        devon2 = filtered_landmarks[i] * img.shape[0]
                        x17, y17 = devon, devon2
                        x18, y18 = devon, devon2
                        cv2.line(img, (x17, y17), (x18, y18), (0, 255, 0), 3)
                        prev_landmarks = filtered_landmarks
            # Update previous landmark positions





            drawColor = (224,190,153)
            brushThickness = 100
            
            y2 = leftIndex_Y*2
            x2 = leftIndex_X*3        
            cv2.circle(img, (x2, y2), 15, (0,0,0), cv2.FILLED)#drawing mode is represented as circle
            #print("Drawing Mode")
            if xp == 0 and yp == 0:#initially xp and yp will be at 0,0 so it will draw a line from 0,0 to whichever point our tip is at
                
                xp, yp = x2, y2 # so to avoid that we set xp=x1 and yp=y1
            #till now we are creating our drawing but it gets removed as everytime our frames are updating so we have to define our canvas where we can draw and show also
                
                if drawColor != (0, 0, 0):
                    cv2.line(img, (xp, yp), (x2, y2), drawColor, brushThickness)#gonna draw lines from previous coodinates to new positions 
                    cv2.line(imgCanvas, (xp, yp), (x2, y2), drawColor, brushThickness)
                xp,yp=x2,y2 # giving values to xp,yp everytime 
                
        # 1 converting img to gray
        imgCanvas = cv2.resize(imgCanvas, [CANV_WIDTH,CANV_HEIGHT])
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        
        # 2 converting into binary image and thn inverting
        _, imgInv = cv2.threshold(imgGray, 30, 255, cv2.THRESH_TRUNC)#on canvas all the region in which we drew is black and where it is black it is cosidered as white,it will create a mask
        
        imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2RGB)#converting again to gray bcoz we have to add in a RGB image i.e img
        
        imgInv = cv2.resize(img, [CANV_WIDTH,CANV_HEIGHT])
        #add original img with imgInv ,by doing this we get our drawing only in black color
        
        img = cv2.bitwise_and(img,imgInv)
        #add img and imgcanvas,by doing this we get colors on img
        # DRAW #
        img = cv2.bitwise_or(img,imgCanvas)
        img = cv2.flip(img, 90)
        ########


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
        cv2.imshow('Pose Detection: Everwave', img)

        if cv2.waitKey(1) == ord('q'):
                break
def HandDetection():
    
    CANV_WIDTH = 1920
    CANV_HEIGHT = 1280

    pTime = 0
    cTime = 0
    # Initialize webcam #
    # Insert 0 for integrated webcam
    # Insert 1 for USB webcam
    cap = cv2.VideoCapture(0)
    imgCanvas = np.zeros((CANV_HEIGHT, CANV_WIDTH, 3), np.uint8)# defining canvas


    wrist = 0
    thumb1 = 1
    thumb2 = 2
    thumb3 = 3
    thumb4 = 4

    pointer1 = 5
    pointer2 = 6
    pointer3 = 7
    pointer4 = 8

    middle1 = 9
    middle2 = 10
    middle3 = 11
    middle4 = 12

    ring1 = 13
    ring2 = 14
    ring3 = 15
    ring4 = 16

    pinky1 = 17
    pinky2 = 18
    pinky3 = 19
    pinky4 = 20

    finger = '2'
    if finger == '1':
        fingerChoice = 4
    elif finger == '2':
        fingerChoice = 8
    elif finger == '3':
        fingerChoice = 12
    elif finger == '4':
        fingerChoice = 16
    elif finger == '5':
        fingerChoice = 20

    while True:

        success, img = cap.read()
        xp, yp = 0, 0
        # Start the hand detector, and run it based off img
        img = detector.findHands(img, draw=False)
        # Find the x,y coordinates of ALL hand landmarks
        lmList = detector.findPosition(img, draw=False)
        
        img = cv2.resize(img, [CANV_WIDTH,CANV_HEIGHT])
    

        # CONSOLE PRINTING OPTIONS #
        if len(lmList) != 0:

            # Assign only the landmarks specified to variables fingerX,Y 
            fingerX = lmList[fingerChoice][1]
            fingerY = lmList[fingerChoice][2]
            
            # you should make this a dbl loop function
            xPalm, yPalm = lmList[0][1], lmList[0][2]
            xPalm1, yPalm1 = lmList[1][1], lmList[1][2]
            xPalm2, yPalm2 = lmList[2][1], lmList[2][2]
            xPalm3, yPalm3 = lmList[3][1], lmList[3][2]
            x1, y1 = lmList[4][1], lmList[4][2]        
            xPointer1, yPointer1 = lmList[5][1], lmList[5][2]
            xPointer2, yPointer2 = lmList[6][1], lmList[6][2]
            xPointer3, yPointer3 = lmList[7][1], lmList[7][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            xMiddle1, yMiddle1 = lmList[9][1], lmList[9][2]
            xMiddle2, yMiddle2 = lmList[10][1], lmList[10][2]
            xMiddle3, yMiddle3 = lmList[11][1], lmList[11][2]
            x3, y3 = lmList[12][1], lmList[12][2]
            xRing1, yRing1 = lmList[13][1], lmList[13][2]
            xRing2, yRing2 = lmList[14][1], lmList[14][2]
            xRing3, yRing3 = lmList[15][1], lmList[15][2]
            x4, y4 = lmList[16][1], lmList[16][2]
            xPinky1, yPinky1 = lmList[17][1], lmList[17][2]
            xPinky2, yPinky2 = lmList[18][1], lmList[18][2]
            xPinky3, yPinky3 = lmList[19][1], lmList[19][2]
            x5, y5 = lmList[20][1], lmList[20][2]



            #########################################
            #hand-off
            clm0 = xPalm
            clm1 = xPalm1
            clm2 = xPalm2
            clm3 = xPalm3
            clm4 = x1
            clm5 = xPointer1
            clm6 = xPointer2
            clm7 = xPointer3
            clm8 = x2
            clm9 = xMiddle1
            clm10 = xMiddle2
            clm11 = xMiddle3
            clm12 = x3
            clm13 = xRing1
            clm14 = xRing2
            clm15 = xRing3
            clm16 = x4
            clm17 = xPinky1
            clm18 = xPinky2
            clm19 = xPinky3
            clm20 = x5
            yclm0 = yPalm
            yclm1 = yPalm1
            yclm2 = yPalm2
            yclm3 = yPalm3
            yclm4 = y1
            yclm5 = yPointer1
            yclm6 = yPointer2
            yclm7 = yPointer3
            yclm8 = y2
            yclm9 = yMiddle1
            yclm10 =yMiddle2
            yclm11 =yMiddle3
            yclm12 =y3
            yclm13 =yRing1
            yclm14 =yRing2
            yclm15 =yRing3
            yclm16 =y4
            yclm17 =yPinky1
            yclm18 =yPinky2
            yclm19 =yPinky3
            yclm20 =y5
            clm0 += 1280
            lemmetry = clm0
            clm1 += 1280
            clm2 += 1280
            clm3 += 1280
            clm4 += 1280
            clm5 += 1280
            clm6 += 1280
            clm7 += 1280
            clm8 += 1280
            clm9 += 1280
            clm10 += 1280
            clm11 += 1280
            clm12 += 1280
            clm13 += 1280
            clm14 += 1280
            clm15 += 1280
            clm16 += 1280
            clm17 += 1280
            clm18 += 1280
            clm19 += 1280
            clm20 += 1280 
            yclm0 +=720
            lemmetry = yclm0
            yclm1 +=720
            yclm2 +=720
            yclm3 +=720
            yclm4 +=720
            yclm5 +=720
            yclm6 +=720
            yclm7 +=720
            yclm8 +=720
            yclm9 +=720
            yclm10 += 720
            yclm11 += 720
            yclm12 += 720
            yclm13 += 720
            yclm14 += 720
            yclm15 += 720
            yclm16 += 720
            yclm17 += 720
            yclm18 += 720
            yclm19 += 720
            yclm20 += 720 
            if yclm0 != yclm0:
                yclm0 = lemmetry

                cv2.circle(img, (clm0, yclm0), 10, (0,0,0), -1)
                cv2.circle(img, (clm4, yclm4), 10, (0,0,0), -1)
                cv2.circle(img, (clm8,yclm8), 10, (0,0,0), -1)
                cv2.circle(img, (clm12,yclm12), 10, (0,0,0), -1)
                # Circle on ring finger
                cv2.circle(img, (clm16, yclm16), 10, (0,0,0), -1)
                # Circle on pinky finger
                cv2.circle(img, (clm20, yclm20), 10, (0,0,0), -1)
            
            #########################################
            # Read cx as center of the two x values on a line
            # Read cy as the center of the two y values on a line

            # Thumb to pointer
            cx,cy = (x1+x2)//2, (y1+y2)//2
            
            # Pointer to middle
            cx1,cy1 = (x2+x3)//2, (y2+y3)//2
            
            # Middle to ring
            cx2,cy2 = (x3+x4)//2, (y3+y4)//2
            
            # Ring to pinky
            cx3,cy3 = (x4+x5)//2, (y4+y5)//2

            # #########################################        
            # # Draw a line connecting thumb and pointer circles
            # cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 3)

            # # Draw a line connecting pointer and middle circles
            # cv2.line(img, (x2, y2), (x3, y3), (255,0,0), 3)

            # # Draw a line connecting middle and ring circles        
            # cv2.line(img, (x3, y3), (x4, y4), (255,0,0), 3)

            # # Draw a line connecting ring and pinky circles
            # cv2.line(img, (x4, y4), (x5, y5), (255,0,0), 3)

            # # # Draw a line connecting pinky and pointer circles
            # # cv2.line(img, (x1, y1), (x5,y5), (255,0,0), 3)

            # # #########################################
            # # Draw a circle at the center (cx,cy) of thumb/pointer
            # cv2.circle(img, (cx,cy), 10, (0,0,255),  -1)
            
            # # Draw a circle at the center (cx,cy) of pointer/middle
            # cv2.circle(img, (cx1,cy1), 10, (0,0,255),  -1)
            
            # # Draw a circle at the center (cx,cy) of middle/ring
            # cv2.circle(img, (cx2,cy2), 10, (0,0,255),  -1)
            
            # # Draw a circle at the center (cx,cy) of ring/pinky
            # cv2.circle(img, (cx3,cy3), 10, (0,0,255),  -1)

            # ##########################################
            # Find the length of the thumb/pointer line
            length = math.hypot(x2-x1, y2-y1)
            
            # Find the length of the pointer/middle line
            length1 = math.hypot(x3-x2, y3-y2)
            
            # Find the length of the middle/ring line
            length2 = math.hypot(x4-x3, y4-y3)
            
            # Find the length of the ring/pinky line
            length3 = math.hypot(x5-x4, y5-y4)

            # Find the length of the thumb and pinky
            length4 = math.hypot(x5-x1, y5-y1)
            
            # If the length is less than 50, turn the red cx,cy dots green
            greenLight = 0
            # I need to like...
            def greenLight():
                    
                # Thumb to Pointer
                if length < 50:
                    cv2.circle(img, (cx,cy), 10, (0,255,0), -1)
                    greenLight += 1
                
                # Pointer to Middle
                if length1 < 50:
                    cv2.circle(img, (cx1,cy1), 10, (0,255,0), -1)
                    greenLight += 2
                
                # Middle to Ring
                if length2 < 50:
                    cv2.circle(img, (cx2,cy2), 10, (0,255,0), -1)
                    greenLight += 3
                
                # Ring to Pinky
                if length3 < 50:
                    cv2.circle(img, (cx3,cy3), 10, (0,255,0), -1)
                    greenLight += 4

                # Hand sign recognition v1.0 ############
                if greenLight == 10:
                    print('Fist')
                    

                elif greenLight == 7:
                    print('Pointer Up')
                
                elif greenLight == 4:
                    print('Peace Sign')
                
                elif greenLight == 1:
                    print('A-OK')

                elif greenLight == 3:
                    print('Horns')
                
                elif greenLight == 9:
                    print('Thumbs Up')
                
                elif greenLight == 0:
                    print('Hi!')
            # Apply moving average filter to smooth out landmark movement
            # Jesus christ holy fuck that was difficult
            # This is so cool
            def MovingAverage():
                prev_landmarks = 50
                filtered_landmarks = []
                for i in range(len(lmList)):
                    if prev_landmarks:
                        filtered_landmarks.append(lmList)
                    else:

                        Filter_ = ((num_frames - 1) * prev_landmarks[i] + lmList[i]) / num_frames
                        filtered_landmark = Filter_
                        filtered_landmarks.append(filtered_landmark)

                # Draw lines between the hand landmarks to create a drawing effect
                for i in range(len(filtered_landmarks)  ):
                    filtered_landmarks[i]
                    if i >= 0:
                            
                        devon= filtered_landmarks[i] * img.shape[1]
                        devon2 = filtered_landmarks[i] * img.shape[0]
                        x17, y17 = devon, devon2
                        x18, y18 = devon, devon2
                        cv2.line(img, (x17, y17), (x18, y18), (0, 255, 0), 3)
                        prev_landmarks = filtered_landmarks
            # Update previous landmark positions





            drawColor = (150,25,150)
            brushThickness = 100
            
            y2 = y2*2
            x2 = x2*3        
            cv2.circle(img, (x2, y2), 15, (0,0,0), cv2.FILLED)#drawing mode is represented as circle
            #print("Drawing Mode")
            if xp == 0 and yp == 0:#initially xp and yp will be at 0,0 so it will draw a line from 0,0 to whichever point our tip is at
                
                xp, yp = x2, y2 # so to avoid that we set xp=x1 and yp=y1
            #till now we are creating our drawing but it gets removed as everytime our frames are updating so we have to define our canvas where we can draw and show also
                
                if drawColor != (0, 0, 0):
                    cv2.line(img, (xp, yp), (x2, y2), drawColor, brushThickness)#gonna draw lines from previous coodinates to new positions 
                    cv2.line(imgCanvas, (xp, yp), (x2, y2), drawColor, brushThickness)
                xp,yp=x2,y2 # giving values to xp,yp everytime 
                
        # 1 converting img to gray
        imgCanvas = cv2.resize(imgCanvas, [CANV_WIDTH,CANV_HEIGHT])
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        
        # 2 converting into binary image and thn inverting
        _, imgInv = cv2.threshold(imgGray, 30, 255, cv2.THRESH_TRUNC)#on canvas all the region in which we drew is black and where it is black it is cosidered as white,it will create a mask
        
        imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2RGB)#converting again to gray bcoz we have to add in a RGB image i.e img
        
        imgInv = cv2.resize(img, [CANV_WIDTH,CANV_HEIGHT])
        #add original img with imgInv ,by doing this we get our drawing only in black color
        
        img = cv2.bitwise_and(img,imgInv)
        #add img and imgcanvas,by doing this we get colors on img
        # DRAW #
        img = cv2.bitwise_or(img,imgCanvas)
        img = cv2.flip(img, 90)
        ########


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
        
        seuss1 = cv2.imread("Motion\Modules\WorkingCodes\FingerDetectionBasics\yg.jpg")
        seuss1_Array = np.array(seuss1)
        print(seuss1_Array.shape)
        cv2.imshow('Hand Detection: Everwave', img)

        if cv2.waitKey(1) == ord('q'):
                break

def handDet():
    print("Start HandDetection:",time.ctime())
    
    triggerHand = True
    if True:
        HandDetection()
        return triggerHand    
def poseDet():
    print("Start PoseDetection:",time.ctime())
    
    # two options to trigger
    triggerPose = True
    
    if True:
        PoseDetection()
    # Name 'PoseDetection is not defined means I have to put this after the functions were defined
    return triggerPose
Pygui()

# these 2 programs share the same variables...
# is that a prob
