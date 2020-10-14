import cv2
import numpy as np
import dlib
from math import hypot

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midPoint(p1,p2):
    return int((p1.x+p2.x)/2), int((p1.y+p2.y)/2)

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:

        # x,y = face.left(), face.top()
        # x1,y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
        #print(face)
        landmarks = predictor(gray, face)
       # print(landmarks)
       #  x = landmarks.part(36).x
       #  y = landmarks.part(36).y
       #  cv2.circle(frame, (x,y), 3, (0,0,255), 2)
        R_leftPoint = (landmarks.part(36).x, landmarks.part(36).y)
        R_rightPoint = (landmarks.part(39).x, landmarks.part(39).y)
        R_centerTop = midPoint(landmarks.part(37), landmarks.part(38))
        R_centerBottom = midPoint(landmarks.part(41), landmarks.part(40))

        # horizontalLine = cv2.line(frame, R_leftPoint, R_rightPoint, (0,0,200), 2)
        # verticalLine = cv2.line(frame, R_centerTop, R_centerBottom, (0,0,200), 2)

        L_leftPoint = (landmarks.part(42).x, landmarks.part(42).y)
        L_rightPoint = (landmarks.part(45).x, landmarks.part(45).y)
        L_centerTop = midPoint(landmarks.part(43), landmarks.part(44))
        L_centerBottom = midPoint(landmarks.part(47), landmarks.part(46))

        # horizontalLine = cv2.line(frame, L_leftPoint, L_rightPoint, (0, 0, 200), 2)
        # verticalLine = cv2.line(frame, L_centerTop, L_centerBottom, (0, 0, 200), 2)

        verticalLine_length = hypot((L_centerTop[0]-L_centerBottom[0]),(L_centerTop[1]-L_centerBottom[1]))
        horizontalLine_length = hypot((L_leftPoint[0]-L_rightPoint[0]),(L_leftPoint[1]-L_rightPoint[1]))
        verticalLine_length_R = hypot((R_centerTop[0] - R_centerBottom[0]), (R_centerTop[1] - R_centerBottom[1]))
        horizontalLine_length_R = hypot((R_leftPoint[0] - R_rightPoint[0]), (R_leftPoint[1] - R_rightPoint[1]))

        ratio = (horizontalLine_length/verticalLine_length)+(verticalLine_length_R/horizontalLine_length_R)

        if ratio>5.5:
            cv2.putText(frame, "BLINK", (50,100), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255),2)
            print("Blink")

    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()