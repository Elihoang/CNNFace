import cv2
import os
import time

# Khởi tạo camera
cam = cv2.VideoCapture(0)

# Tải bộ phát hiện khuôn mặt Haar Cascade
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Lấy thông tin người dùng
face_id = input('\n Đảm bảo người dùng đầu tiên nhập vào là 0. Nhập mã số người dùng và nhấn <return> ==>  ')
face_name = input(f'\n Nhập tên cho {face_id} <return> ==>  ')

print("\n [THÔNG TIN] Đang khởi tạo chụp khuôn mặt. Nhìn vào camera và chờ đợi ...")
count = 0

# Tạo thư mục dataset nếu chưa có
dataset_dir = 'datasets/new_persons'
user_dir = os.path.join(dataset_dir, face_id)

if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)

if not os.path.exists(user_dir):
    os.mkdir(user_dir)

# Biến tính toán FPS
frame_count = 0
start_time = time.time()
fps = 0  # Khởi tạo biến FPS

# Quay video
while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Lật ảnh video theo chiều dọc
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Duyệt qua tất cả các khuôn mặt phát hiện được
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w + 50, y + h + 50), (255, 0, 0), 2)

        # Cắt và lưu ảnh khuôn mặt vào thư mục của người dùng
        face_image = gray[y:y + h, x:x + w]

        # Kiểm tra nếu khuôn mặt không bị rỗng trước khi lưu
        if face_image.size > 0:  # Kiểm tra nếu ảnh không rỗng
            try:
                path_save_face = os.path.join(user_dir, f"User.{face_id}.{count}.jpg")
                cv2.imwrite(path_save_face, face_image)
            except Exception as e:
                print(f"Lỗi khi lưu ảnh khuôn mặt: {e}")
                continue
        else:
            print(f"Cảnh báo: Vùng khuôn mặt rỗng được phát hiện tại chỉ số {count}")

        # Tính toán FPS và hiển thị lên ảnh
        frame_count += 1
        if frame_count >= 1:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, f"anh da chup: {count}", (img.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('image', img)
        count += 1

    # Dừng video khi nhấn 'ESC' hoặc chụp đủ 70 ảnh
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif count >= 150:
        break

# Dọn dẹp và thoát chương trình
print("\n [THÔNG TIN] Đang thoát chương trình và dọn dẹp")
cam.release()
cv2.destroyAllWindows()
