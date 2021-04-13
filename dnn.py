###############################################
# Derek Mease
# CSCI 4831-5722 (Fleming)
# Final Project - Computer Vision Photo Sorter
###############################################

# This file contains code for filtering images using pre-trained deep neural networks.
# Code implemented using pretrained weights from the following:
#   https://github.com/pjreddie/darknet
#   https://gilberttanner.com/blog/yolo-object-detection-with-opencv

import numpy as np
import cv2 as cv
from os.path import join


# Draw bounding boxes for detected objects.
# Params:
#   image - the image to draw on
#   boxes - bounding boxes returned from DNN
#   confidences - confidence values fro each bounding box
#   idxs - indexes for the list of bounding boxes
#   label - label for the object detected
def draw_boxes(image, boxes, confidences, idxs, label):
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Scale bounding box
            height, width, _ = image.shape
            box = boxes[i] * np.array([width, height, width, height])
            centerX, centerY, w, h = box.astype('int')
            x = int(centerX - (w / 2))
            y = int(centerY - (h / 2))

            # Draw bounding box
            color = (0, 0, 255)  # red
            box_thickness = max(1, int(1 * width // 500))
            font_thickness = max(1, int(0.5 * width // 500))
            font_size = max(0.5, 0.5 * width // 1000)
            cv.rectangle(image, (x, y), (x + w, y + h), color, box_thickness)
            text = "{}: {:.4f}".format(label, confidences[i])
            cv.putText(image, text, (x, y - 5 - box_thickness),
                        cv.FONT_HERSHEY_SIMPLEX, font_size, color,
                        font_thickness)

    return image


# Class to detect objects with YOLO DNN
class Yolo:
    def __init__(self, confidence=0.5, threshold=0.3):
        # Load pretrained weights and labels
        self.labels = open(
            'data/dnn/yolo/config/coco.names').read().strip().split('\n')
        self.net = cv.dnn.readNetFromDarknet(
            'data/dnn/yolo/config/yolov3.cfg', 'data/dnn/yolo/config/yolov3.weights')

        # Ouput layer names
        self.layer_names = self.net.getLayerNames()
        self.layer_names = [self.layer_names[i[0] - 1]
                            for i in self.net.getUnconnectedOutLayers()]

        self.confidence = confidence
        self.threshold = threshold

    # Detect objects in an image
    def detect(self, label, image):
        blob = cv.dnn.blobFromImage(
            image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.layer_names)

        boxes = []
        confidences = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                conf = scores[classID]

                # Keep only bounding boxes for the label we are looking for
                # with confidence over the threshold.
                if self.labels[classID] == label and conf > self.confidence:
                    boxes.append(detection[0:4])
                    confidences.append(float(conf))

        idxs = cv.dnn.NMSBoxes(
            boxes, confidences, self.confidence, self.threshold)

        return boxes, confidences, idxs

    # Filter out images that contain an oject with the label
    def filter(self, path, files, label, progress):
        filtered_images = []
        detections = []

        for f in files:
            progress.set_status(f'Detecting {label} in {f}')

            file_path = join(path, f)
            image = cv.imread(file_path)
            boxes, confidences, idxs = self.detect(label, image)
            print(f'{f}: {len(idxs)} {label}s found')
            if len(idxs) > 0:
                filtered_images.append(f)
                detections.append(self.Detection(
                    f, boxes, confidences, idxs, label))
            progress.increment(1)

        return filtered_images, detections

    # Utility class to store detection information.
    class Detection:
        def __init__(self, image, boxes, confidences, idxs, label):
            self.image = image
            self.boxes = boxes
            self.confidences = confidences
            self.idxs = idxs
            self.label = label
