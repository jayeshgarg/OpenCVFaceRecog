import cv2
import numpy as np
import pickle

from pip._vendor.distlib.compat import raw_input

# face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

cap = cv2.VideoCapture(0)
try:
    while True:

        name = raw_input("Enter you name before training : ")

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                # Draw blue box to show what we are learning
                color = (255, 0, 0)  # BGR 0-255
                stroke = 2
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

                roi = gray[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(current_id)  # as my id in other trainer is 1

            cv2.imshow('frame', frame)
            print(str(current_id) + " : " + name)
            if cv2.waitKey(20) & 0xFF == ord('x'):
                print({current_id: name})
                label_ids[name] = current_id
                current_id += 1
                break

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
except AttributeError:
    print(label_ids)
# print(y_labels)
# print(x_train)
print(label_ids)
# with open("pickles/face-labels" + trainer + ".pickle", 'wb') as f:
with open("pickles/face-labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
# recognizer.save("recognizers/face-trainner" + trainer + ".yml")
recognizer.save("recognizers/face-trainner.yml")
