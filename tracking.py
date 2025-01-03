import time

import cv2
import yaml

from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking


# Hàm tải tệp cấu hình YAML
def load_config(file_name):
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


# Hàm thực hiện phát hiện và theo dõi đối tượng
def inference(detector, args):
    # Mở đối tượng video capture
    cap = cv2.VideoCapture(0)

    # Khởi tạo các biến để đo tốc độ khung hình
    start_time = time.time_ns()
    frame_count = 0
    fps = -1

    # Khởi tạo bộ theo dõi và bộ đếm thời gian
    tracker = BYTETracker(args=args, frame_rate=30)
    frame_id = 0

    while True:
        # Đọc một khung hình từ video capture
        ret_val, frame = cap.read()

        if ret_val:
            # Thực hiện phát hiện và theo dõi khuôn mặt trên khung hình
            outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)

            if outputs is not None:
                online_targets = tracker.update(
                    outputs, [img_info["height"], img_info["width"]], (128, 128)
                )
                online_tlwhs = []
                online_ids = []
                online_scores = []

                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
                    if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)

                online_im = plot_tracking(
                    img_info["raw_img"],
                    online_tlwhs,
                    online_ids,
                    frame_id=frame_id + 1,
                    fps=fps,
                )
            else:
                online_im = img_info["raw_img"]

            # Tính toán và hiển thị tốc độ khung hình
            frame_count += 1
            if frame_count >= 30:
                fps = 1e9 * frame_count / (time.time_ns() - start_time)
                frame_count = 0
                start_time = time.time_ns()

            # Vẽ hộp giới hạn và các điểm mốc trên khung hình
            # for i in range(len(bboxes)):
            #     # Lấy vị trí của khuôn mặt
            #     x1, y1, x2, y2, score = bboxes[i]
            #     cv2.rectangle(online_im, (x1, y1), (x2, y2), (200, 200, 230), 2)

            cv2.imshow("Face Tracking", online_im)

            # Kiểm tra đầu vào thoát của người dùng
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1


def main():
    file_name = "./face_tracking/config/config_tracking.yaml"
    config_tracking = load_config(file_name)
    # detector = Yolov5Face(
    #     model_file="face_detection/yolov5_face/weights/yolov5m-face.pt"
    # )
    detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

    inference(detector=detector, args=config_tracking)


if __name__ == "__main__":
    main()
