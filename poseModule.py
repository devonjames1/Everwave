import cv2
import mediapipe as mp
import time

class poseDetector():

    def __init__(self, mode = False, 
                 upperBody = False, 
                 smooth = True, 
                 detectionCon = True, 
                 trackingCon = True):
        
        self.mode = mode
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,
                                     self.upperBody,
                                     self.smooth,
                                     self.detectionCon,
                                     self.trackingCon)
    
    def findPose(self, img, draw = True):
        # Converet to RGB
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, 
                                        self.results.pose_landmarks, 
                                        self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        
        lmList = []

        if self.results.pose_landmarks:        
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                # print(id,'\n', lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:    
                    cv2.circle(img, (cx, cy), 10, (200,67,100), cv2.FILLED, 4)
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    while True:

        success,img = cap.read() 
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw = False)
        if len(lmList) != 0:
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 10, (200,67,100), cv2.FILLED, 4)
            print(lmList[1])
        
        cTime = time.time()
        fps = 1/(cTime-pTime)    
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70,90), cv2.FONT_HERSHEY_COMPLEX, 3,
                    (255,255,255), 1)
        
        cv2.imshow('Hey', img)    
        cv2.waitKey(1)



if __name__ == '__main__':
    main()