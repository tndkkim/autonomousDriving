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
    #    if not grabbed:
    #        print("No input image")
    #        break

        frame = imutils.resize(frame, width=500)
        frameArea = frame.shape[0]*frame.shape[1]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 사각형 인식을 위한 사전 처리
        mask = preprocessForRectangleDetection(hsv)

        # 사각형 및 해당하는 표지판 탐지
        detectedTrafficSign, largestRect = detectRectangleAndClassifySign(mask, frame, frameArea)

        # 결과 표시
        if largestRect is not None:
            cv2.drawContours(frame, [largestRect], 0, (0, 255, 0), 2)
            warped = four_point_transform(frame, largestRect.reshape(4, 2))
            detectedTrafficSign = identifyTrafficSign(warped)

            # 표지판이 "Unknown"으로 식별되었다면, 흑백 이미지에서 재식별을 시도합니다.
            if detectedTrafficSign == "Unknown":
                # 워핑된 흑백 이미지 생성
                warped2 = four_point_transform(mask, largestRect.reshape(4, 2))
                detectedTrafficSign = identifyTrafficSign2(warped2)
            
            # 결과가 여전히 "Unknown"이 아닌 경우에만 표시
            if detectedTrafficSign != "Unknown":
                cv2.putText(frame, detectedTrafficSign, tuple(largestRect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.imshow("Original", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    camera.release()
    cv2.destroyAllWindows()

def preprocessForRectangleDetection(hsv):
    # 넓은 범위의 색상 마스크를 생성하여 대부분의 색상을 포함시킵니다.
    # 이는 사각형 형태를 더 잘 감지하기 위함입니다.
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower, upper)

    # 모폴로지 연산으로 노이즈 제거
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
        # 가장 큰 사각형을 찾습니다.
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
        for cnt in cnts:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)  # 수정된 부분

            sideOne = np.linalg.norm(box[0]-box[1])
            sideTwo = np.linalg.norm(box[0]-box[3])
                # count area of the rectangle
            area = sideOne*sideTwo
                # find the largest rectangle within all contours
            if area > largestArea:
                largestArea = area
                largestRect = box

            
            # 사각형 영역의 색상 분석을 위해 투영 변환 적용
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
    # 각 색상 범위 정의 (널널하게 설정)
    yellow = cv2.inRange(hsv, (20, 80, 80), (40, 255, 255))

    lower_red = cv2.inRange(hsv, (0, 90, 30), (10, 255, 255))
    upper_red = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    red = cv2.bitwise_or(lower_red, upper_red)

    #green = cv2.inRange(hsv, (50, 100, 100), (70, 255, 255))

    white = cv2.inRange(hsv, (0, 0, 200), (180, 25, 255))
    black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))


    #kernel = np.ones((3, 3), np.uint8)
    #red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
    #red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)

        # 특정 색상 픽셀 수 계산
    yellow_pixels = cv2.countNonZero(yellow)
    #red_pixels = cv2.countNonZero(red)
    #green_pixels = cv2.countNonZero(green)
    white_pixels = cv2.countNonZero(white)
    black_pixels = cv2.countNonZero(black)

    print(white_pixels)

        # 표지판 식별 로직
    if yellow_pixels > 300:  # yellow sign
        if yellow_pixels > 450 and white_pixels < 3000:
            return "School1"
        else:
            return "School2"

    contours, _ = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area > 100:  # 너무 작은 윤곽선은 무시
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

            if len(approx) == 3:  # 삼각형
                return "Tunnel"
            elif len(approx) > 5:  # 원은 근사 윤곽선이 많은 점으로 구성됨
                #(x, y), radius = cv2.minEnclosingCircle(cnt)
                #center = (int(x), int(y))
                #radius = int(radius)
                #if radius > 10:  # 너무 작은 원은 무시
                return "Stop"

                
        


    return "Unknown"

def identifyTrafficSign2(edges):
    # Hough line transform to find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    

    height, width = edges.shape
    left_half = edges[:, :width // 2]
    right_half = edges[:, width // 2:]
    cnts_left, _ = cv2.findContours(left_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_right, _ = cv2.findContours(right_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Initialize angle ranges for left, straight, and right
    #left_angles = []
    #right_angles = []


    lengths_left = sum([cv2.arcLength(cnt, True) for cnt in cnts_left])
    lengths_right = sum([cv2.arcLength(cnt, True) for cnt in cnts_right])

    #print(lengths_left, lengths_right)

    # Decide the direction based on the majority of angles

    if lengths_left >= 120 + lengths_right:
        return "Left"
    elif lengths_right >= 120 + lengths_left:
        return "Right"
    else:
        return "Straight"
    
    return "Unknown"
    


# 메인 함수 실행
if __name__ == '__main__':
    findTrafficSign()