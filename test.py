import cv2
import numpy as np
from keras.models import load_model
import pygame

# Inisialisasi pygame
pygame.mixer.init()

# Load pre-trained Haar Cascade for face and eyes
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load your deep learning model
model_path = "Model.h5"  # Ganti dengan path model yang sudah Anda latih
new_model = load_model(model_path)

cap = cv2.VideoCapture(0)

# Load sound file
sound_path = "test1.mp3"  # Ganti dengan path file suara Anda
pygame.mixer.music.load(sound_path)

# Set threshold untuk menentukan status mata
threshold = 0.5

while cap.isOpened():
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(face_roi)
        
        for (ex, ey, ew, eh) in eyes:
            eyes_roi = face_roi[ey:ey+eh, ex:ex+ew]
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

            # Preprocess the eyes_roi for your deep learning model
            final_image = cv2.resize(eyes_roi, (224, 224 ))
            final_image = np.expand_dims(final_image, axis=0)
            final_image = final_image / 255.0

            # Make prediction using the deep learning model
            predictions = new_model.predict(final_image)

            # Display the prediction result
            status = "Open Eyes" if predictions > threshold else "Closed Eyes"
            
            # Play sound if eyes are closed
            if predictions <= threshold:
                pygame.mixer.music.play(-1)  # -1 untuk memutar suara secara terus-menerus
            else:
                pygame.mixer.music.stop()

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, status, (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_4)

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
