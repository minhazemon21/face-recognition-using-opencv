import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'imageAttendence'
image = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImage = cv2.imread(f'{path}/{cl}')
    image.append(curImage)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncoder(image):
    encodelist = []
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist
encodelistknow = findEncoder(image)
print('Encoding Complete')

def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        myDatalist = f.readlines()
        #print(myDatalist)
        nameList = []
        for line in myDatalist:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S:%D')
           # dtString = str(datetime.datetime.now())
            #dateString = now.date('%Y:%M:%D')
            f.writelines(f'\n{name},{dtString}')


cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodelistknow,encodeFace)
        faceDis = face_recognition.face_distance(encodelistknow,encodeFace)
        print(faceDis)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            #cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,name,(x1+5,y2+40),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)

    cv2.imshow("webcam",img)
    cv2.waitKey(1)
