import time
import cv2

from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face

# Khởi tạo bộ phát hiện khuôn mặt
#detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5m-face.pt")
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")


def main():
    # Mở camera
    cap = cv2.VideoCapture(0)

    # Khởi tạo các biến để đo tốc độ khung hình
    start = time.time_ns()
    frame_count = 0
    fps = -1

    # Lưu video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    video = cv2.VideoWriter(
        "results/face-detection.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, size
    )

    # Đọc các khung hình từ camera
    while True:
        # Chụp một khung hình từ camera
        _, frame = cap.read()

        # Lấy các khuôn mặt và các điểm đặc trưng từ bộ phát hiện khuôn mặt
        bboxes, landmarks = detector.detect(image=frame)
        h, w, c = frame.shape

        tl = 1 or round(0.002 * (h + w) / 2) + 1  # Độ dày đường viền và phông chữ
        clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]  # Màu sắc cho các điểm đặc trưng

        # Vẽ các hộp giới hạn và điểm đặc trưng lên khung hình
        for i in range(len(bboxes)):
            # Lấy vị trí của khuôn mặt
            x1, y1, x2, y2, score = bboxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)

            # Vẽ các điểm đặc trưng trên khuôn mặt
            for id, key_point in enumerate(landmarks[i]):
                cv2.circle(frame, tuple(key_point), tl + 1, clors[id], -1)

        # Tính toán và hiển thị tốc độ khung hình
        frame_count += 1
        if frame_count >= 30:
            end = time.time_ns()
            fps = 1e9 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(
                frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        # Lưu khung hình vào video
        video.write(frame)

        # Hiển thị kết quả trong cửa sổ
        cv2.imshow("Face Detection", frame)

        # Nhấn 'Q' trên bàn phím để thoát
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    # Giải phóng video và camera, và đóng tất cả cửa sổ OpenCV
    video.release()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
