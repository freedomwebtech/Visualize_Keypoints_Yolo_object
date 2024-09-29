import cv2
import numpy as np
import cvzone

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

# Initial positions for the cars
car1_y = 390  # Position of car 1 (lane 1)
car2_y = 390  # Position of car 2 (lane 2)

while True:
    # Create a copy of the road image for each frame
    img_result = road_image.copy()
    
    # Overlay the cars on their respective lanes
    img_result = cvzone.overlayPNG(img_result, car_image, [lane1_center_x - car_image.shape[1] // 2, car1_y])
    img_result = cvzone.overlayPNG(img_result, car1_image, [lane2_center_x - car_image.shape[1] // 2, car2_y])
    
    # Draw lane center circles for visualization
    cv2.circle(img_result, (lane1_center_x, 490), 10, (0, 255, 0), -1)  # Lane 1 center
    cv2.circle(img_result, (lane2_center_x, 490), 10, (0, 0, 255), -1)  # Lane 2 center
    
    # Show the frame
    cv2.imshow("frame", img_result)

    key = cv2.waitKey(1)
    
    # Control the car movement with keyboard input
    if key == ord('w'):  # Move car 1 up
        car1_y -= 5
    elif key == ord('s'):  # Move car 1 down
        car1_y += 5
    elif key == ord('i'):  # Move car 2 up
        car2_y -= 5
    elif key == ord('k'):  # Move car 2 down
        car2_y += 5
    elif key == ord('q'):  # Quit
        break

# Clean up
cv2.destroyAllWindows()
