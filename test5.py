import cv2
import numpy as np
import cvzone
from ultralytics import YOLO

# Load COCO class names
with open("coco.txt", "r") as f:
    class_names = f.read().splitlines()

# Load the YOLOv8 model
model = YOLO("yolov10n.pt")

# Open the video file
cap = cv2.VideoCapture('1.mov')

# Load the car image
car_image = cv2.imread('mycar.png', cv2.IMREAD_UNCHANGED)
car_image = cv2.resize(car_image, (50, 50))
car1_image = car_image.copy()  # Duplicate of the car image for the second car

# Load the road image
road_image = cv2.imread('road.png')
road_image = cv2.resize(road_image, (1020, 500))

# Lane centers
lane1_center_x = int(1020 * 0.3)  # 30% of the road width
lane2_center_x = int(1020 * 0.7)  # 70% of the road width

# Dictionary to keep track of car positions by track ID
car_positions = {}
count = 0

while True:
    # Create a copy of the road image for each frame
    img_result = road_image.copy()
    ret, frame = cap.read()
    
    if not ret:
        break

    frame = cv2.resize(frame, (1029, 600))
    
    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True)

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score
        
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            if 'car' in c:
                x1, y1, x2, y2 = box
                cx = int(x1 + x2) // 2
                cy = int(y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (x1, y2), 4, (255, 0, 0), -1)
                cvzone.putTextRect(frame, f'ID: {track_id}', (x1, y1), 1, 1)

                # Update car position in the dictionary
                car_positions[track_id] = cy
                print(cx,lane1_center_x)   
                # Overlay the cars on their respective lanes using track ID
                if track_id in car_positions:
                    # Determine which lane to use based on the x-coordinate of the car
                    if cx < lane1_center_x:  # If the car is in the first lane
                       img_result = cvzone.overlayPNG(img_result, car_image, [lane1_center_x - car_image.shape[1] // 2, car_positions[track_id]])
                       cvzone.putTextRect(img_result, f'Track ID: {track_id}', 
                                           (lane1_center_x - car_image.shape[1] // 2, car_positions[track_id] - 10), 1, 1)
                    else:  # If the car is in the second lane
                       img_result = cvzone.overlayPNG(img_result, car1_image, [lane2_center_x - car1_image.shape[1] // 2, car_positions[track_id]])
                       cvzone.putTextRect(img_result, f'Track ID: {track_id}', 
                                           (lane2_center_x - car1_image.shape[1] // 2, car_positions[track_id] - 10), 1, 1)

    # Show the frames
    cv2.imshow("frame", img_result)
    cv2.imshow("frame1", frame)

    if cv2.waitKey(0) & 0xFF == ord("q"):
        break
  
# Clean up
cap.release()
cv2.destroyAllWindows()
