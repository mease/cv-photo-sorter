###############################################
# Derek Mease
# CSCI 4831-5722 (Fleming)
# Final Project - Computer Vision Photo Sorter
###############################################

# This file contains code for filtering images using HAAR cascades.

import cv2 as cv
from os.path import join

# Filter list of images that contain detections from the cascade classifier matching a label.
# Params:
#   path - path to image files
#   files - image file names
#   label - the object to find. Must have a "label.xml" cascade file in ./data/cascade/.
#   progress - the progress bar
def filter(path, files, label, progress):
    hits = []
    detections_list = []

    cascade_file = join('./data/cascade/', label + '.xml')
    cascade = cv.CascadeClassifier(cascade_file)

    for f in files:
        progress.set_status(f'Detecting {label} in {f}')
        file_path = join(path, f)
        gray = cv.imread(file_path, cv.IMREAD_GRAYSCALE)

        # Resize large images for faster processing
        max_width = 2000
        height, width = gray.shape
        if width > max_width:
            scale_percent = max_width / width
            width = int(width * scale_percent)
            height = int(height * scale_percent)
            dim = (width, height)
            gray = cv.resize(gray, dim, interpolation=cv.INTER_AREA)
        else:
            scale_percent = 1

        # Get detections
        detections = cascade.detectMultiScale(gray)
        print(f'{f}: {len(detections)} detections')

        # Resize bounding boxes to match the original image size
        resized_detections = []
        for d in detections:
            resized_detections.append(tuple([int(x/scale_percent) for x in d]))
        detections = resized_detections

        # Save file name with detections
        if len(detections) > 0:
            hits.append(f)
            detections_list.append(detections)
        progress.increment(1)

    return hits, detections_list

# Draw bounding boxes on an image
# Params:
#   img - the image to draw on
#   boxes - detections returned from cascade classifier
def draw_boxes(img, boxes):
    height, width, _ = img.shape
    color = (0, 0, 255)  # red
    box_thickness = max(1, int(1 * width // 500))

    for (x,y,w,h) in boxes:
        cv.rectangle(img, (x, y), (x + w, y + h), color, box_thickness)

    return img
