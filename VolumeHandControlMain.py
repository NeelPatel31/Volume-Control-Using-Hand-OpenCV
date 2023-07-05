import cv2 as cv
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

######################
wCam, hCam = 640,480
######################

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]



cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime=0

detector = htm.handDetector(min_detection_confidence=0.7)

vol = 0
volBar = 400
volPer = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img,True)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList)!=0:
        # print(lmList[4],lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx,cy = (x1+x2)//2,(y1+y2)//2

        cv.circle(img,(x1,y1),15,(255,0,255),cv.FILLED)
        cv.circle(img,(x2,y2),15,(255,0,255),cv.FILLED)
        cv.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv.circle(img,(cx,cy),15,(255,0,255),cv.FILLED)

        length = math.hypot(x2-x1,y2-y1)
        # print(length)

        #  Hand range 10 to 215
        # Volume range -95 to 0
        vol = np.interp(length,[10,215],[minVol,maxVol])
        volBar = np.interp(length,[10,215],[400,150])
        volPer = np.interp(length,[10,215],[0,100])

        print(int(length),vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv.circle(img,(cx,cy),15,(0,255,0),cv.FILLED)

    cv.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),-1)
    cv.putText(img,f"{int(volPer)}%",(40,450),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img,f"FPS: {int(fps)}",(20,50),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    cv.imshow("Img",img)

    if cv.waitKey(1) & 0xFF==ord('d'):
        exit(0)