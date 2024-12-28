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
cv2.namedWindow("Lip and Eye Effects")
cv2.createTrackbar("Blue", "Lip and Eye Effects", 0, 255, empty_function)
cv2.createTrackbar("Green", "Lip and Eye Effects", 0, 255, empty_function)
cv2.createTrackbar("Red", "Lip and Eye Effects", 0, 255, empty_function)

# Function to apply fisheye effect to the eyes
def barrel(src, k):
    w = src.shape[1]
    h = src.shape[0]

    # Meshgrid of destination image
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Normalize x and y
    x = np.float32(x) / w - 0.5
    y = np.float32(y) / h - 0.5

    # Radial distance from center
    r = np.sqrt(np.square(x) + np.square(y))

    # Implementing the barrel distortion equation
    dr = np.multiply(k * r, np.cos(np.pi * r))

    # Outside the maximum radius dr is set to 0
    dr[r > 0.5] = 0

    # Apply the distortion
    rn = r - dr

    # Remap the pixels
    xd = cv2.divide(np.multiply(rn, x), r)
    yd = cv2.divide(np.multiply(rn, y), r)

    # Convert back to original coordinates
    xd = w * (xd + 0.5)
    yd = h * (yd + 0.5)

    # Apply remapping to the source image
    dst = cv2.remap(src, xd, yd, cv2.INTER_CUBIC)
    return dst

# Function to apply fisheye effect to the eyes
def apply_fisheye(img, landmarks, bulgeAmount=0.75, radius=30):
    # Find the ROI for both eyes
    roiEyeLeft = [landmarks.part(37).x - radius,
                  landmarks.part(37).y - radius,
                  landmarks.part(40).x - landmarks.part(37).x + 2 * radius,
                  landmarks.part(41).y - landmarks.part(37).y + 2 * radius]

    roiEyeRight = [landmarks.part(43).x - radius,
                   landmarks.part(43).y - radius,
                   landmarks.part(46).x - landmarks.part(43).x + 2 * radius,
                   landmarks.part(47).y - landmarks.part(43).y + 2 * radius]

    output = np.copy(img)

    # Apply fisheye effect on left eye
    leftEyeRegion = img[roiEyeLeft[1]:roiEyeLeft[1] + roiEyeLeft[3], roiEyeLeft[0]:roiEyeLeft[0] + roiEyeLeft[2]]
    leftEyeRegionDistorted = barrel(leftEyeRegion, bulgeAmount)
    output[roiEyeLeft[1]:roiEyeLeft[1] + roiEyeLeft[3], roiEyeLeft[0]:roiEyeLeft[0] + roiEyeLeft[2]] = leftEyeRegionDistorted

    # Apply fisheye effect on right eye
    rightEyeRegion = img[roiEyeRight[1]:roiEyeRight[1] + roiEyeRight[3], roiEyeRight[0]:roiEyeRight[0] + roiEyeRight[2]]
    rightEyeRegionDistorted = barrel(rightEyeRegion, bulgeAmount)
    output[roiEyeRight[1]:roiEyeRight[1] + roiEyeRight[3], roiEyeRight[0]:roiEyeRight[0] + roiEyeRight[2]] = rightEyeRegionDistorted

    return output

# Function to smooth skin for face region only
def smooth_skin_face(image, landmarks):
    # Create a mask for the face region
    face_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(0, 17)], dtype=np.int32)
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(mask, [face_points], (255, 255, 255))

    # Apply bilateral filter to the face region only
    smoothed = cv2.bilateralFilter(image, d=15, sigmaColor=75, sigmaSpace=75)
    face_only = cv2.bitwise_and(smoothed, mask)
    background = cv2.bitwise_and(image, cv2.bitwise_not(mask))

    # Combine face region with the rest of the image
    return cv2.add(face_only, background)

# Function to apply eye color to the pupil only
def change_pupil_color(image, landmarks, color=(0, 255, 0)):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Left eye (pupil) points
    left_eye_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in [37, 38, 40, 41]], np.int32)
    cv2.fillPoly(mask, [left_eye_points], 255)

    # Right eye (pupil) points
    right_eye_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in [43, 44, 46, 47]], np.int32)
    cv2.fillPoly(mask, [right_eye_points], 255)

    # Apply color only to the masked region (pupil)
    eye_color = np.zeros_like(image)
    eye_color[:, :] = color

    # Blur for smooth blending
    eye_color = cv2.bitwise_and(eye_color, eye_color, mask=mask)
    eye_color = cv2.GaussianBlur(eye_color, (7, 7), 5)

    # Blend the eye color with the original image
    return cv2.addWeighted(image, 1, eye_color, 0.6, 0)

# Function to add blush effect on the cheeks
def add_blush(image, landmarks, color=(0, 0, 255), intensity=0.5, radius=30):
    # Define the points for cheeks (near landmarks 2, 14)
    left_cheek_center = (landmarks.part(2).x, landmarks.part(2).y)
    right_cheek_center = (landmarks.part(14).x, landmarks.part(14).y)

    # Create a transparent overlay for the blush effect
    overlay = image.copy()

    # Draw blush circles on the cheeks
    cv2.circle(overlay, left_cheek_center, radius, color, -1)
    cv2.circle(overlay, right_cheek_center, radius, color, -1)

    # Blend the overlay with the original image
    return cv2.addWeighted(overlay, intensity, image, 1 - intensity, 0)

# Start the webcam capture
cap = cv2.VideoCapture(0)

# Main loop to apply lip color filter, fisheye effect, skin smoothing, and blush effect
current_eye_color = None  # Start with original eye color
while True:
    # Grab the current frame from the webcam
    ret, image = cap.read()

    # If the frame is not read correctly, break the loop
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the image horizontally
    image = cv2.flip(image, 1)

    original_image = image.copy()

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_image)

    # Loop through all detected faces
    for face in faces:
        # Detect facial landmarks
        landmarks = landmark_predictor(gray_image, face)

        # Smooth the skin for the face region only
        image = smooth_skin_face(image, landmarks)

        # Change eye color only for pupils
        if current_eye_color is not None:
            image = change_pupil_color(image, landmarks, color=current_eye_color)

        # Apply fisheye effect on eyes
        image = apply_fisheye(image, landmarks, bulgeAmount=0.75, radius=30)

        # Add blush effect to the cheeks
        image = add_blush(image, landmarks, color=(0, 0, 255), intensity=0.6, radius=40)

        # Extract the lip region from the facial landmarks
        lips_region = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(48, 61)], np.int32)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [lips_region], (255, 255, 255))

        lips_color = np.zeros_like(image)
        blue_value = cv2.getTrackbarPos('Blue', 'Lip and Eye Effects')
        green_value = cv2.getTrackbarPos('Green', 'Lip and Eye Effects')
        red_value = cv2.getTrackbarPos('Red', 'Lip and Eye Effects')
        lips_color[:, :] = (blue_value, green_value, red_value)
        lips_color = cv2.bitwise_and(lips_color, lips_color, mask=mask)
        lips_color = cv2.GaussianBlur(lips_color, (7, 7), 10)

        image = cv2.addWeighted(image, 1, lips_color, 0.4, 0)

    # Display the final output with the effects
    cv2.imshow('Lip and Eye Effects', image)

    # Check for user input to change eye color or capture the image
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        cv2.imwrite('captured_image.jpg', image)
        print("Image captured and saved as captured_image.jpg")
    elif key == ord('g'):
        current_eye_color = (34, 139, 34)  # Green
        print("Eye color set to Green")
    elif key == ord('b'):
        current_eye_color = (255, 0, 0)  # Blue
        print("Eye color set to Blue")
    elif key == ord('r'):
        current_eye_color = None  # Reset to original eye color
        print("Eye color reset to original")

    # Exit the loop when 'q' is pressed
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
