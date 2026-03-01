import os
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import time
from tracker import Tracker
import matplotlib.pyplot as plt

# Configuration
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.chdir(r"C:\Users\sudhi\Desktop\Shafaque")

# Constants for line positions and settings
BLUE_LINE_Y = 150        # Upper line Y position (start timer)
RED_LINE_Y = 200         # Lower line Y position (stop timer)
OFFSET = 10              # Margin of error for line crossing
DISTANCE_METERS = 50     # Real distance between lines in meters (adjust for your scenario)
MAX_SPEED_THRESHOLD = 5 # Speed limit km/h
# CLASS_LIST = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
#               'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
#               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
#               'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
#               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#               'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#               'toothbrush']

# Initialize YOLO model and tracker
# yolo = YOLO(r"C:\Users\sudhi\Desktop\Shafaque\shafaque.pt")
yolo = YOLO("yolov5s.pt")
# print(yolo.names)
tracker = Tracker()
# number_detector = Number()  # Uncomment and implement plate detection if needed

# Video source
cap = cv2.VideoCapture(r"202VehicleDetection\20250527_124108_F1CC_B8A44FD03ADD.mkv")

# Dictionary to store time when vehicle crosses upper line
object_times = {}
# List of vehicle IDs that have been counted (to avoid double count)
counter_up = []

def process_frame(frame):
    # Run YOLO prediction on frame
    result = yolo.predict(frame)
    detections = result[0].boxes.data.cpu().numpy()
    # Return DataFrame with detection boxes and class IDs
    # print(detections)
    return pd.DataFrame(detections, columns=['x1', 'y1', 'x2', 'y2', 'conf', 'class'])

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    detections = process_frame(frame)
    cars = detections[detections['class'].isin(
        [2,3,5,7]
        # [0,1,2,3]
    )]
    objects = cars[['x1', 'y1', 'x2', 'y2']].values.tolist()

    tracked_objects = tracker.update(objects)

    for x1, y1, x2, y2, obj_id in tracked_objects:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # Debug: print object center and ID
        print(f"ID {obj_id}: center y={cy}")
        label = f"ID {obj_id}"
        cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),   # Green color (BGR)
                    2              # Thickness
                )
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # Background rectangle
        # cv2.rectangle(
        #     frame,
        #     (x1, y1 - th - 15),
        #     (x1 + tw + 5, y1),
        #     (0, 255, 0),
        #     -1
        # )

        # Text
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )


        # Start timing when crossing blue line (upper)
        if BLUE_LINE_Y - OFFSET < cy < BLUE_LINE_Y + OFFSET:
            if obj_id not in object_times:
                object_times[obj_id] = time.time()
                print(f"ID {obj_id} crossed BLUE line at {object_times[obj_id]}")
                count+=1

        # Calculate speed when crossing red line (lower)
        if obj_id in object_times and RED_LINE_Y - OFFSET < cy < RED_LINE_Y + OFFSET:
            elapsed_time = time.time() - object_times[obj_id]
            if elapsed_time <= 0:
                # Safety check: elapsed_time should not be zero or negative
                print(f"Warning: non-positive elapsed_time for ID {obj_id}")
                continue

            speed_kmh = (DISTANCE_METERS / elapsed_time) * 3.6
            label = f" {speed_kmh:.1f} km/h"

            cv2.putText(
                frame,
                label,
                (x1+50, y1 - 5),           # above bounding box
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            print(f"ID {obj_id} crossed RED line. Elapsed time: {elapsed_time:.2f}s Speed: {speed_kmh:.2f} km/h")
            

            if obj_id not in counter_up and speed_kmh > MAX_SPEED_THRESHOLD:
                counter_up.append(obj_id)
                print(f"Speeding car detected! ID {obj_id} speed: {speed_kmh:.2f} km/h")
                # plate_img = frame[y1:y2, x1:x2]
                # plate_number = number_detector.plate(plate_img)
                # if plate_number:
                #     print(f"Plate: {plate_number}")

    # Draw lines
    cv2.line(frame, (8, BLUE_LINE_Y), (927, BLUE_LINE_Y), (255, 0, 0), 2)  # blue
    cv2.line(frame, (8, RED_LINE_Y), (927, RED_LINE_Y), (0, 0, 255), 2)     # red

    cv2.putText(frame, f"Speeding Vehicle: {len(counter_up)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f"Counting of Vehicle: {count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow('Speed Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
