import cv2
import os
import numpy as np

dataPath = './data'
peopleList = os.listdir(dataPath)

labels = []
faceData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath+'/'+nameDir
    print("leyendo imagenes")

    for filename in os.listdir(personPath):
        labels.append(label)
        faceData.append(cv2.imread(personPath+'/'+filename,0))
        image = cv2.imread(personPath+'/'+filename,0)

    label += 1

face_recognizer = cv2.face.EigenFaceRecognizer()

# Entrenando reconocer de rostros
print("entrenando")
face_recognizer.train(faceData,np.array(labels))

#almacenando el modelo obtenido
face_recognizer.write('modeloEigenFace.xml')
print('almacenando modelo obtenido')


    