import cv2 as cv
import numpy as np
import random

# Load the video or webcam
cap = cv.VideoCapture('Input_Video/image_to_video.avi')

whT = 320
confThreshold = 0.3  # Lower threshold to detect more objects
nmsThreshold = 0.4   # Increase to allow more overlapping boxes

# Load class names
classesFile = "Crime/crime.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model configuration and weights
modelConfiguration = "Crime/crime.cfg"
modelWeights = "Crime/crime.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox, classIds, confs = [], [], []

    # Loop through detections
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            # Apply threshold
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                
                # Store detections
                bbox.append([x, y, w, h])
                classIds.append(classId)

                # Introduce randomness to confidence (for debugging)
                confs.append(float(confidence * random.uniform(0.8, 1.2)))

    # Apply Non-Maximum Suppression (NMS)
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    # Ensure indices exist to prevent errors
    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box

            # Draw bounding box
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # Display class name & confidence
            label = f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%'
            cv.putText(img, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


while True:
    success, img = cap.read()
    if not success:
        break

    # Prepare input for YOLO
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    # Get model output layers
    layersNames = net.getLayerNames()
    outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    outputs = net.forward(outputNames)

    # Process detections
    findObjects(outputs, img)

    # Resize for display
    imgg = cv.resize(img, (960, 540))
    cv.imshow('Image', imgg)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
