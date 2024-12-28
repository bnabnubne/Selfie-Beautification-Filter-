import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Help functions:

def drawPolyline(im, landmarks, start, end, color, thickness=1, isClosed=False):
    points = [[landmarks.part(i).x, landmarks.part(i).y] for i in range(start, end + 1)] 
    points = np.array(points, dtype=np.int32)
    cv2.polylines(im, [points], isClosed, color, thickness=thickness, lineType=cv2.LINE_8)

def barrel(src, k):
    w = src.shape[1]
    h = src.shape[0]

    # Meshgrid of destiation image
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Normalize x and y
    x = np.float32(x) / w - 0.5
    y = np.float32(y) / h - 0.5

    # Radial distance from center
    r = np.sqrt(np.square(x) + np.square(y))

    # Implementing the following equaition
    # dr = k * r * cos(pi * r)
    dr = np.multiply(k * r, np.cos(np.pi * r))

    # Outside the maximum radius dr is set to 0
    dr[r > 0.5] = 0

    # Remember we need to provide inverse mapping to remap
    # Hence the negative sign before dr
    rn = r - dr

    # Applying the distortion on the grid
    xd = cv2.divide(np.multiply(rn, x), r)
    yd = cv2.divide(np.multiply(rn, y), r)

    # Back to un-normalized coordinates
    xd = w * (xd + 0.5)
    yd = h * (yd + 0.5)

    # Apply warp to source image using remap
    dst = cv2.remap(src, xd, yd, cv2.INTER_CUBIC)
    return dst

# Facial Filters function:

# Filter 1- Lipstick
def apply_lipstick(image, landmarks, color=(0, 0, 255)):
    # Convert landmarks to a numpy array (example: using the jawline part)
    upper_lip_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(48, 55)] +  # Upper part
                                [[landmarks.part(i).x, landmarks.part(i).y] for i in range(61, 64)], dtype=np.int32).reshape((1, -1, 2))  # Lower part
    cv2.fillPoly(image, [upper_lip_points], color)

    # Points for the lower lip: includes the outer and inner parts of the lip
    lower_lip_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(54, 61)] +  # Upper part
                                [[landmarks.part(i).x, landmarks.part(i).y] for i in range(66, 68)], dtype=np.int32).reshape((-1, 1, 2))  # Lower part
    cv2.fillPoly(image, [lower_lip_points], color)

    return image

# Filter 2 - Eyeliners
def apply_eyeliners(image, landmarks, thickness: int = 2):
    color = (0, 0, 0)

    drawPolyline(image, landmarks, 36, 41, color, thickness, True)  # Left Eye
    drawPolyline(image, landmarks, 42, 47, color, thickness, True)  # Right Eye

# Filter 3 - FishEyes
def apply_fisheye(img, landmark, bulgeAmount:float = 0.75, radius: int = 30):
    # Control the amount of deformation.

    # Find the roi for left Eye
    roiEyeLeft = [landmark.part(37).x - radius,
                  landmark.part(37).y - radius,
                  landmark.part(40).x - landmark.part(37).x + 2 * radius,
                  landmark.part(41).y - landmark.part(37).y + 2 * radius]

    # Find the roi for right Eye
    roiEyeRight = [landmark.part(43).x - radius,
                   landmark.part(43).y - radius,
                  landmark.part(46).x - landmark.part(43).x + 2 * radius,
                  landmark.part(47).y - landmark.part(43).y + 2 * radius]

    output = np.copy(img)

    # Find the patch for left eye and apply the transformation
    leftEyeRegion = img[roiEyeLeft[1]:roiEyeLeft[1] + roiEyeLeft[3], roiEyeLeft[0]:roiEyeLeft[0] + roiEyeLeft[2]]
    leftEyeRegionDistorted = barrel(leftEyeRegion, bulgeAmount)
    output[roiEyeLeft[1]:roiEyeLeft[1] + roiEyeLeft[3], roiEyeLeft[0]:roiEyeLeft[0] + roiEyeLeft[2]] = leftEyeRegionDistorted

    # Find the patch for right eye and apply the transformation
    rightEyeRegion = img[roiEyeRight[1]:roiEyeRight[1] + roiEyeRight[3], roiEyeRight[0]:roiEyeRight[0] + roiEyeRight[2]]
    rightEyeRegionDistorted = barrel(rightEyeRegion, bulgeAmount)
    output[roiEyeRight[1]:roiEyeRight[1] + roiEyeRight[3], roiEyeRight[0]:roiEyeRight[0] + roiEyeRight[2]] = rightEyeRegionDistorted

    return output

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    while True:
        # Grab Image from camera:
        ret, frame = cap.read()

        # If cannot got an image:
        if not ret:
            break

        # Convert Image to 1 channel ( Gray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect Faces im image:
        faces = detector(gray)

        # For each face:
        for face in faces:

            # Calculate face landmarks:
            landmarks = predictor(gray, face)

            # Apply lipstick:
            apply_lipstick(frame, landmarks)

            # Apply eyeliners:
            apply_eyeliners(frame, landmarks)

            # Apply fisheye:
            apply_fisheye(frame, landmarks, bulgeAmount= 0.75, radius = 30)

        # Show image:
        cv2.imshow('Virtual Makeup', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Close camera when done
    cap.release()

    # Close image windows:
    cv2.destroyAllWindows()