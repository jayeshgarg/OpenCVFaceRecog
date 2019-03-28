# import the necessary packages
import os
import cv2
import pickle
import time
import imutils
import numpy as np

from keras.models import load_model
from imutils.video import VideoStream

detector = "face_detector"
model = "model/liveness.model"
le = "pickles/le.pickle"
min_confidence = 0.5

recognizer = cv2.face.LBPHFaceRecognizer_create()

protoPath = os.path.sep.join([detector, "deploy.prototxt"])
modelPath = os.path.sep.join([detector, "res10_300x300_ssd_iter_140000.caffemodel"])

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

label_ids = {"person_name": 1}
y_labels = []
x_train = []

with open(le, 'rb') as f:
    og_labels = pickle.load(f)
    label_ids = {v: k for k, v in og_labels.items()}

net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] loading liveness detector...")
model = load_model(model)

vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            roi = gray[startY:endY, startX:endX]

            id_, conf = recognizer.predict(roi)

            if 4 <= conf:
                # print(5: #id_)
                # print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = label_ids[id_]
                color = (0, 255, 255)
                stroke = 2
                label = "{}: {:.4f}".format(name, conf)
                cv2.putText(frame, label, (startX, startY), font, 1, color, stroke, cv2.LINE_AA)
            # show the output frame and wait for a key press
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
