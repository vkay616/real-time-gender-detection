import numpy as np
import cv2
import cvlib as cv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# LOADING THE TRAINED MODEL
model = load_model('trained_model.model')

# TO ENABLE REALTIME CAPTURE USING WEBCAM
webcam = cv2.VideoCapture(0)

classes = ['male', 'female']

# LOOPING THROUGH THE FACES (REALTIME)
while webcam.isOpened():

    status, frame = webcam.read()

    faces, confidence = cv.detect_face(frame)

    for index, face in enumerate(faces):

        (X_start, y_start) = face[0], face[1]
        (X_end, y_end) = face[2], face[3]

        # CREATING A RECTANGULAR BORDER AROUND THE FACE
        cv2.rectangle(frame, (X_start, y_start), (X_end, y_end), (0,255,0), 2)

        # CROPPING THE FACE
        face_crop = np.copy(frame[y_start:y_end, X_start:X_end])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue
        
        # PRE-PROCESSING THE CROPPED FACE IMAGE
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype('float') / 255
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # USING THE TRAINED MODEL TO PREDICT THE GENDER
        predictor = model.predict(face_crop)[0]

        index = np.argmax(predictor)
        label = classes[index]

        # GENDER AND THE ACCURACY OF THE MODEL
        label = "{}: {:.2f}%".format(label, predictor[index]*100)

        y = y_start - 10 if y_start - 10 > 10 else y_start + 10

        # SHOWING THE PREDICTED GENDER AND THE ACCURACY
        cv2.putText(frame, label, (X_start, y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow('Gender Detector', frame)

    # PRESS ESC KEY TO CLOSE THE PROGRAM
    if cv2.waitKey(100) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
