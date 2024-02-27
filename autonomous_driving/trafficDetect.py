import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
camera = cv2.VideoCapture(0)
camera.set(3,320) 
camera.set(4,240)

#camera =

def findTrafficSign():
    #pass
    while True:
        grabbed, frame = camera.read()
        frame = imutils.resize(frame, width=500)
        frameArea = frame.shape[0]*frame.shape[1]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = preprocessForRectangleDetection(hsv)

        detectedTrafficSign, largestRect = detectRectangleAndClassifySign(mask, frame, frameArea)

        if largestRect is not None:
            cv2.drawContours(frame, [largestRect], 0, (0, 255, 0), 2)
            warped = four_point_transform(frame, largestRect.reshape(4, 2))
            detectedTrafficSign = identifyTrafficSign(warped)

            if detectedTrafficSign == "Unknown":
                warped2 = four_point_transform(mask, largestRect.reshape(4, 2))
                detectedTrafficSign = identifyTrafficSign2(warped2)
            if detectedTrafficSign != "Unknown":
                cv2.putText(frame, detectedTrafficSign, tuple(largestRect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.imshow("Original", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    camera.release()
    cv2.destroyAllWindows()

def preprocessForRectangleDetection(hsv):
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def detectRectangleAndClassifySign(mask, frame, frameArea):
    detectedTrafficSign = None
    largestRect = None
    largestArea = 0

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
        for cnt in cnts:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)  # 수정된 부분

            sideOne = np.linalg.norm(box[0]-box[1])
            sideTwo = np.linalg.norm(box[0]-box[3])
            area = sideOne*sideTwo

            if area > largestArea:
                largestArea = area
                largestRect = box

            
            warped = four_point_transform(frame, box.reshape(4, 2))
            detectedTrafficSign = identifyTrafficSign(warped)
            break

        if largestArea <= frameArea*0.02:
                largestArea = None

    return detectedTrafficSign, largestRect

#lower_blue = np.array([85,100,70])
#upper_blue = np.array([115,255,255])

def identifyTrafficSign(roi):
    approx = '0'

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    yellow = cv2.inRange(hsv, (20, 80, 80), (40, 255, 255)) #수정필요

    lower_red = cv2.inRange(hsv, (0, 90, 30), (10, 255, 255))
    upper_red = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    red = cv2.bitwise_or(lower_red, upper_red)

    #green = cv2.inRange(hsv, (50, 100, 100), (70, 255, 255))

    white = cv2.inRange(hsv, (0, 0, 200), (180, 25, 255))
    black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))


    #kernel = np.ones((3, 3), np.uint8)
    #red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
    #red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)

    yellow_pixels = cv2.countNonZero(yellow)
    #red_pixels = cv2.countNonZero(red)
    #green_pixels = cv2.countNonZero(green)
    white_pixels = cv2.countNonZero(white)
    #black_pixels = cv2.countNonZero(black)

    print(white_pixels)

    if yellow_pixels > 300:  
        if yellow_pixels > 450 and white_pixels < 3000:
            return "School1"
        else:
            return "School2"

    contours, _ = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area > 100: 
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

            if len(approx) == 3: 
                return "Tunnel"
            elif len(approx) > 5:  
                #(x, y), radius = cv2.minEnclosingCircle(cnt)
                #center = (int(x), int(y))
                #radius = int(radius)
                #if radius > 10: 
                return "Stop"

                
        


    return "Unknown"

def identifyTrafficSign2(edges):
    height, width = edges.shape
    
    right_third = edges[:, :width // 3]
    middle_third = edges[:, width // 3: 2 * width // 3]
    left_third = edges[:, 2 * width // 3:]
    
    white_pixels_left = cv2.countNonZero(left_third)
    white_pixels_middle = cv2.countNonZero(middle_third)
    white_pixels_right = cv2.countNonZero(right_third)

    #print("left : ", white_pixels_left)
    #print("straight : ", white_pixels_middle)
    #print("right : ", white_pixels_right)

    
    if white_pixels_left > white_pixels_middle and white_pixels_left > white_pixels_right:
        return "Left"
    elif white_pixels_middle > white_pixels_left and white_pixels_middle > white_pixels_right:
        return "Straight"
    elif white_pixels_right > white_pixels_left and white_pixels_right > white_pixels_middle:
        return "Right"
    
    return "Unknown"
    


if __name__ == '__main__':
    findTrafficSign()