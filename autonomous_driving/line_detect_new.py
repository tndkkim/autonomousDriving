import cv2 
import numpy as np 
import time
import socket
import sys
import motor
import servo
from choeumpa import distance
from trafficDetect import findTrafficSign


def main():
    camera = cv2.VideoCapture(0)
    camera.set(3,320) 
    camera.set(4,240)
    
    speed=30
    motor.Forward(speed)
    servo.Go()

    while( camera.isOpened() ):
        ret, frame = camera.read()
        frame = cv2.flip(frame,1)
        cv2.imshow('normal',frame)

        traffic_sign = findTrafficSign(frame)
        print("Detected Traffic Sign: ", traffic_sign)
        
        if traffic_sign == "Stop":
            motor.Stop()
            continue

        if traffic_sign == "school":
            motor.GO(speed/3)

        #표지판별로 만들기
        
        crop_img =frame[120:240, 0:320]
        
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        ret,thresh1 = cv2.threshold(blur,130,255,cv2.THRESH_BINARY_INV)
        
        mask = cv2.erode(thresh1, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow('mask',mask)
    
        contours,hierarchy = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)

            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            if cx >= 200 and cx <= 300:              
                print("Left")
                servo.HardTurnleft()
                motor.Left(speed)
            elif cx >= 40 and cx <= 110:
                print("Right")
                servo.Turnright()
                motor.Right(speed)
            else:
                print("go")
                servo.Go()

        dist = distance()
        print("Measured Distance = %.1f cm" % dist)

        if dist > 7:
            print("Forward")
            
        else:
            print("Stop")
            motor.Stop()

            
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    GPIO.cleanup()
