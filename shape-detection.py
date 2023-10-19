# Detecting whether an image contains a rectangle, square, or circle 
# First, make sure you have OpenCV installed: "pip install opencv-python"


import cv2
import numpy as np

def detect_shapes(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use the Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
    
    if circles is not None:
        for circle in circles[0]:
            (x, y, r) = circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
    
    # Detect contours for rectangles and squares
    _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
    
    # Display the result
    cv2.imshow("Shape Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Provide the path to the image you want to analyze
image_path = "path_to_your_image.jpg"
detect_shapes(image_path)
