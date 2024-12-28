import cv2
import numpy as np
import dlib

# Initialize the face detector and facial landmark predictor using dlib
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function for trackbars, currently a placeholder
def empty_function(x):
    pass

# Create a window and trackbars to adjust the RGB values for lip color
cv2.namedWindow("Lip Color Adjuster")
cv2.createTrackbar("Blue", "Lip Color Adjuster", 0, 255, empty_function)
cv2.createTrackbar("Green", "Lip Color Adjuster", 0, 255, empty_function)
cv2.createTrackbar("Red", "Lip Color Adjuster", 0, 255, empty_function)

# Function to extract the region of interest (lips in this case)
def extract_region(image, points, resize_factor=8, mask_area=False, crop_area=True):
    if mask_area:
        # Create a mask for the selected region (e.g., lips)
        mask = np.zeros_like(image)
        mask = cv2.fillPoly(mask, [points], (255, 255, 255))
        image = cv2.bitwise_and(image, mask)
    if crop_area:
        # Crop the region of interest (e.g., the lips area)
        x, y, w, h = cv2.boundingRect(points)
        cropped_image = image[y:y+h, x:x+w]
        cropped_image = cv2.resize(cropped_image, (0, 0), None, resize_factor, resize_factor)
        return cropped_image
    else:
        return mask

# Start the webcam capture
cap = cv2.VideoCapture(0)

# Main loop to apply lip color filter on the webcam feed
while True:
    # Grab the current frame from the webcam
    ret, image = cap.read()

    # If the frame is not read correctly, break the loop
    if not ret:
        print("Failed to grab frame")
        break

    original_image = image.copy()

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_image)

    # Loop through all detected faces
    for face in faces:
        # Detect facial landmarks
        landmarks = landmark_predictor(gray_image, face)
        points = []

        # Store the landmark points in a list
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            points.append([x, y])

        points = np.array(points)

        # Extract the lip region from the facial landmarks
        lips_region = extract_region(image, points[48:61], mask_area=True, crop_area=False)

        # Prepare a blank image for coloring the lips
        lips_color = np.zeros_like(lips_region)

        # Get the RGB values from the trackbars
        blue_value = cv2.getTrackbarPos('Blue', 'Lip Color Adjuster')
        green_value = cv2.getTrackbarPos('Green', 'Lip Color Adjuster')
        red_value = cv2.getTrackbarPos('Red', 'Lip Color Adjuster')

        # Apply the selected color to the lips
        lips_color[:] = blue_value, green_value, red_value
        lips_color = cv2.bitwise_and(lips_region, lips_color)
        lips_color = cv2.GaussianBlur(lips_color, (7, 7), 10)

        # Convert the original image to grayscale and back to color (for blending)
        grayscale_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        grayscale_original = cv2.cvtColor(grayscale_original, cv2.COLOR_GRAY2BGR)

        # Blend the colored lips with the original image
        final_image = cv2.addWeighted(original_image, 1, lips_color, 0.4, 0)

        # Display the final output with the colored lips
        cv2.imshow('Lip Color Adjuster', final_image)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
