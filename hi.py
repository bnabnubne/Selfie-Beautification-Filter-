import cv2
import dlib
import numpy as np

# Khởi tạo Dlib
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Các biến toàn cục
current_effect = None  # Hiệu ứng hiện tại
effects = ["Negative", "Sepia", "Fisheye", "Gray", "Cool","Add Glasses", "Add Dog Nose","Reset"]

# Hàm hiệu ứng

def apply_negative(image):
    return 255 - image

def apply_sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]], dtype=np.float32)
    image = np.array(image, dtype=np.float32) / 255.0
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image * 255, 0, 255).astype(np.uint8)
    return sepia_image

def apply_fisheye(image):
    h, w = image.shape[:2]
    K = np.array([[w, 0, w // 2], [0, h, h // 2], [0, 0, 1]], dtype=np.float32)
    D = np.array([0.7, -0.5, 0.0, 0.0], dtype=np.float32)  # Tăng độ méo để hiệu ứng rõ hơn
    map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_32FC1)
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

def apply_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_cool(image):
    decrease = np.array([0, 10, 20], dtype=np.uint8)
    cool_image = cv2.subtract(image, decrease)
    return np.clip(cool_image, 0, 255)

def add_glasses(image, landmarks):
    glasses = cv2.imread("glasses.png", -1)  # PNG image with transparency
    if glasses is None:
        print("Glasses image not found!")
        return image

    # Tăng tỉ lệ kính bằng scale_factor
    scale_factor = 1.5  # Điều chỉnh giá trị này để làm kính lớn hơn hoặc nhỏ hơn
    glasses_width = int((landmarks.part(45).x - landmarks.part(36).x) * scale_factor)
    glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])

    # Resize kính theo chiều rộng và chiều cao tính toán
    glasses = cv2.resize(glasses, (glasses_width, glasses_height))

    # Tính tọa độ trên-trái kính dựa trên mắt trái (landmark 36)
    top_left = (landmarks.part(36).x - int((glasses_width - (landmarks.part(45).x - landmarks.part(36).x)) / 2),
                landmarks.part(36).y - glasses_height // 2)

    # Kiểm tra nếu vùng kính nằm ngoài ảnh để tránh lỗi
    if (top_left[1] + glasses_height > image.shape[0] or
            top_left[0] + glasses_width > image.shape[1] or
            top_left[0] < 0 or top_left[1] < 0):
        print("Glasses exceed image boundaries")
        return image

    # Gắn kính vào khuôn mặt bằng alpha blending
    for c in range(0, 3):
        roi = image[top_left[1]:top_left[1] + glasses_height, top_left[0]:top_left[0] + glasses_width, c]
        alpha = glasses[:, :, 3] / 255.0
        image[top_left[1]:top_left[1] + glasses_height, top_left[0]:top_left[0] + glasses_width, c] = (
        glasses[:, :, c] * alpha + roi * (1 - alpha)
    )


    return image



def add_dog_nose(image, landmarks):
    nose_image = cv2.imread("dog_nose.png", cv2.IMREAD_UNCHANGED)
    nose_width = (landmarks.part(35).x - landmarks.part(31).x) * 5
    nose_image = cv2.resize(nose_image, (nose_width, nose_width))
    x = landmarks.part(30).x - nose_width // 2
    y = landmarks.part(30).y - nose_width // 2
    for i in range(nose_image.shape[0]):
        for j in range(nose_image.shape[1]):
            if nose_image[i, j, 3] > 0:
                image[y + i, x + j] = nose_image[i, j, :3]
    return image

# Hàm xử lý sự kiện chuột
def on_mouse(event, x, y, flags, param):
    global current_effect
    if event == cv2.EVENT_LBUTTONDOWN:
        # Kiểm tra nếu chuột nằm trong vùng của nút
        for i, effect in enumerate(effects):
            if 10 <= x <= 150 and 50 + i * 50 <= y <= 90 + i * 50:
                current_effect = effect
                print(f"Selected Effect: {effect}")

# Vẽ nút bấm
def draw_buttons(frame):
    for i, effect in enumerate(effects):
        cv2.rectangle(frame, (10, 50 + i * 50), (150, 90 + i * 50), (200, 200, 200), -1)
        cv2.putText(frame, effect, (20, 80 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Main loop
cap = cv2.VideoCapture(0)
cv2.namedWindow("Selfie Beautification Filter")
cv2.setMouseCallback("Selfie Beautification Filter", on_mouse)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count = 0 
    frame = cv2.flip(frame, 1)
    original_frame = frame.copy()  # Giữ lại bản gốc
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_frame)
    landmarks = None 

    # Áp dụng hiệu ứng dựa trên lựa chọn
    if current_effect == "Negative":
        frame = apply_negative(frame)
    elif current_effect == "Sepia":
        frame = apply_sepia(frame)
    elif current_effect == "Fisheye":
        frame = apply_fisheye(frame)
    elif current_effect == "Gray":
        frame = cv2.cvtColor(apply_gray(frame), cv2.COLOR_GRAY2BGR)
    elif current_effect == "Cool":
        frame = apply_cool(frame)

    elif current_effect == "Reset":
        frame = original_frame
        
    if frame_count % 5 == 0:
        faces = face_detector(gray_frame)
        if len(faces) > 0:
            landmarks = landmark_predictor(gray_frame, faces[0])
    # Áp dụng hiệu ứng yêu cầu landmarks
    if landmarks and current_effect == "Add Glasses":
        frame = add_glasses(frame, landmarks)
    elif landmarks and current_effect == "Add Dog Nose":
        frame = add_dog_nose(frame, landmarks)
        
    # Vẽ các nút bấm
    draw_buttons(frame)

    # Hiển thị kết quả
    cv2.imshow("Selfie Beautification Filter", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
