import math
import time
import cv2 as cv
import mediapipe as mp

class handDetector():
    def __init__(self,
               static_image_mode=False,
               max_num_hands=2,
               model_complexity=1,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        
        self.mode = static_image_mode
        self.maxHands = max_num_hands
        self.modelComplex = model_complexity
        self.detectionCon = min_detection_confidence
        self.trackCon = min_tracking_confidence
        
        self.mpHands = mp.solutions.hands
        # by default --> hands = mpHands.Hands(static_image_mode=False,max_num_hands=2,model_complexity=1,min_detection_confidence=0.5,min_detection_confidence=0.5)
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelComplex,self.detectionCon,self.trackCon)
        # hands = self.mpHands.Hands(False,2,1,0.5,0.5)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]

    def findHands(self,img,draw=True):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw = True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx,cy)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy),10,(255,0,255),-1)
        return self.lmList

    def fingersUp(self):
        fingers= []

        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # 4 fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self,p1,p2,img,draw=True,radius=15,t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx,cy = (x1+x2)//2,(y1+y2)//2

        if draw:
            cv.line(img,(x1,y1),(x2,y2),(255,0,255),thickness=t)
            cv.circle(img,(x1,y1),radius,(255,0,255),-1)
            cv.circle(img,(x2,y2),radius,(255,0,255),-1)
            cv.circle(img,(cx,cy),radius,(0,0,255),-1)
        length = math.hypot(x2-x1,y2-y1)

        return length,img,[x1,y1,x2,y2,cx,cy]


def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)

    detector = handDetector()
    while True:
        success,img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,3,(0,255,0),3)
        cv.imshow("Image",img)
        
        if cv.waitKey(20) & 0xFF==ord('d'):
            break

if __name__ == "__main__":
    main()
