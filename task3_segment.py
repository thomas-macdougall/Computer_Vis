import cv2
import numpy as np

import pickle
import os

# Resize each segment detected for an image
def resize_segments(segments, size=(95, 95)):
    resized = []
    for segment in segments:
        resized_segment = cv2.resize(segment.copy(), size, interpolation=cv2.INTER_AREA)
        resized.append(resized_segment)
    return resized

# Segment each image in images by icons
def segment_images(images, resize=False, save=False):
    # if file already exists then unpickle
    if os.path.isfile("./segmented_task3_tests.pkl"):
        with open("./segmented_task3_tests.pkl", "rb") as f:
            segmented_icons_list = pickle.load(f)
            return segmented_icons_list
    
    # co ordinates of segmented icons for each image
    co_ords_segments = []

    segmented_icons_list = []
    counter = 1
    for o_img in images:
        print(f"Segmenting test images ({counter}/ {len(images)})", end="\r")
        # Preprocess image
        img = o_img.copy()
        o_w, o_h = img.shape[0], img.shape[1]
        img = cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
        blur = cv2.medianBlur(img, 25)
        _, thresh = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((11, 11), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Compute bounding box for each detected contour
        icon_bounding_boxes = []
        for cnt in contours:
            x, y, w, h = [round(dim / 10) for dim in cv2.boundingRect(cnt)]
            
            if w * h < 500:
                continue
            x_min, y_min = max(x - 1, 0), max(y - 1, 0)
            x_max, y_max = min(x + w + 1, o_h), min(y + h + 1, o_w)
            
            icon_bounding_boxes.append((x_min, y_min, x_max, y_max))

        # Filter out bounding boxes that are covered
        filtered_boxes = []
        for i, (x1, y1, x2, y2) in enumerate(icon_bounding_boxes):
            covering = True
            for j, (x3, y3, x4, y4) in enumerate(icon_bounding_boxes):
                if i == j: continue
                if x3 <= x1 and x4 >= x2 and y3 <= y1 and y4 >= y2:
                    covering = False
                    break
            if covering:
                filtered_boxes.append((x1, y1, x2, y2))

        # Pad image segments with borders
        image_segments = []
        for (x_min, y_min, x_max, y_max) in filtered_boxes:
            image_segment = o_img[y_min:y_max, x_min:x_max].copy()
            padded_segment = cv2.copyMakeBorder(
                image_segment,
                top=5,
                bottom=5,
                left=5,
                right=5,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255)
            )
            image_segments.append(padded_segment)
        segmented_icons_list.append(image_segments)

        # co ordinates of the segments
        co_ords_segments.append(filtered_boxes)
        counter += 1

    # Resize image segments to a single fixed size
    if resize:
        resized = []
        for segments in segmented_icons_list:
            resized.append(resize_segments(segments=segments))
        segmented_icons_list = resized
    
    if save:
        with open("./segmented_task3_tests.pkl", "wb") as f:
            pickle.dump(zip(segmented_icons_list, co_ords_segments), f)
    
    return zip(segmented_icons_list, co_ords_segments)

        
