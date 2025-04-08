import cv2
import os
import imutils

personName = 'example'
dataPath = './data'
personPath= dataPath+'/'+personName

if not os.path.exists(personPath):
    print("A directory was created: ",personPath)
    os.makedirs(personPath)

cap = cv2.VideoCapture('./db-videos/example.mp4')

faceClassIf = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 0

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame,width=380, height=420)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassIf.detectMultiScale(gray,1.4,5)    
    for (x,y,w,h) in faces:
        
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
        face2 = auxFrame[y:y+h,x:x+w]
        face = cv2.resize(face2,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count),face)
        count = count +1
    cv2.imshow('frame',frame)

    k= cv2.waitKey(1)
    if k == 27 or count >= 300:
        break

cap.release()
cv2.destroyAllWindows()