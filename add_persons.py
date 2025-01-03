import argparse
import os
import shutil
import cv2
import numpy as np
import torch
from torchvision import transforms
from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import read_features

# Kiểm tra xem CUDA có sẵn không và đặt thiết bị phù hợp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo bộ phát hiện khuôn mặt (Chọn một trong các bộ phát hiện)
# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

# Khởi tạo bộ nhận diện khuôn mặt
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)


@torch.no_grad()
def get_feature(face_image):
    """
    Extract facial features from an image using the face recognition model.

    Args:
        face_image (numpy.ndarray): Input facial image.

    Returns:
        numpy.ndarray: Extracted facial features.
    """
    # Định nghĩa các bước tiền xử lý ảnh
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Chuyển đổi ảnh sang định dạng RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Áp dụng các bước tiền xử lý lên ảnh
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    # Sử dụng mô hình để trích xuất đặc trưng khuôn mặt
    emb_img_face = recognizer(face_image)[0].cpu().numpy()

    # Normalize the features
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)
    return images_emb


def add_persons(backup_dir, add_persons_dir, faces_save_dir, features_path):
    """
    Add a new person to the face recognition database.

    Args:
        backup_dir (str): Directory to save backup data.
        add_persons_dir (str): Directory containing images of the new person.
        faces_save_dir (str): Directory to save the extracted faces.
        features_path (str): Path to save face features.
    """
    # Khởi tạo danh sách để lưu tên và đặc trưng của các ảnh đã thêm
    images_name = []
    images_emb = []

    # Đọc thư mục chứa ảnh của người mới, phát hiện khuôn mặt, và lưu lại
    for name_person in os.listdir(add_persons_dir):
        person_image_path = os.path.join(add_persons_dir, name_person)

        # Tạo thư mục để lưu khuôn mặt của từng người
        person_face_path = os.path.join(faces_save_dir, name_person)
        os.makedirs(person_face_path, exist_ok=True)

        for image_name in os.listdir(person_image_path):
            if image_name.endswith(("png", "jpg", "jpeg")):
                input_image = cv2.imread(os.path.join(person_image_path, image_name))

                # Phát hiện khuôn mặt và các điểm đặc trưng sử dụng bộ phát hiệ
                bboxes, landmarks = detector.detect(image=input_image)

                # Trích xuất khuôn mặt
                for i in range(len(bboxes)):
                    # Lấy số lượng file hiện có trong thư mục của người đó
                    number_files = len(os.listdir(person_face_path))

                    # Lấy tọa độ khuôn mặt
                    x1, y1, x2, y2, score = bboxes[i]

                    # Đảm bảo vùng khuôn mặt hợp lệ
                    if x1 < x2 and y1 < y2:
                        # Cắt khuôn mặt từ ảnh
                        face_image = input_image[y1:y2, x1:x2]

                        # Kiểm tra xem ảnh khuôn mặt có hợp lệ không
                        if face_image.size > 0:
                            # Đường dẫn lưu khuôn mặt
                            path_save_face = os.path.join(person_face_path, f"{number_files}.jpg")

                            # Lưu khuôn mặt vào cơ sở dữ liệu
                            cv2.imwrite(path_save_face, face_image)

                            # Trích xuất đặc trưng từ khuôn mặt
                            images_emb.append(get_feature(face_image=face_image))
                            images_name.append(name_person)
                        else:
                            print(f"Cảnh báo: Khuôn mặt rỗng trong {image_name}")
                    else:
                        print(f"Cảnh báo: Tọa độ không hợp lệ trong {image_name}")

    # Kiểm tra nếu không có người mới
    if images_emb == [] and images_name == []:
        print("Không tìm thấy người mới!")
        return None

    # Chuyển đổi danh sách thành mảng
    images_emb = np.array(images_emb)
    images_name = np.array(images_name)

    # Đọc đặc trưng hiện có nếu có
    features = read_features(features_path)

    if features is not None:
        # Giải nén đặc trưng hiện có
        old_images_name, old_images_emb = features

        # Gộp đặc trưng mới với đặc trưng cũ
        images_name = np.hstack((old_images_name, images_name))
        images_emb = np.vstack((old_images_emb, images_emb))

        print("Cập nhật đặc trưng!")

        # Lưu đặc trưng đã gộp
    np.savez_compressed(features_path, images_name=images_name, images_emb=images_emb)

    # Di chuyển dữ liệu của người mới vào thư mục sao lưu
    for sub_dir in os.listdir(add_persons_dir):
        dir_to_move = os.path.join(add_persons_dir, sub_dir)
        shutil.move(dir_to_move, backup_dir, copy_function=shutil.copytree)

    print("Thêm người mới thành công!")


if __name__ == "__main__":
    # Phân tích các tham số dòng lệnh
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="./datasets/backup",
        help="Directory to save person data.",
    )
    parser.add_argument(
        "--add-persons-dir",
        type=str,
        default="./datasets/new_persons",
        help="Directory to add new persons.",
    )
    parser.add_argument(
        "--faces-save-dir",
        type=str,
        default="./datasets/data/",
        help="Directory to save faces.",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="./datasets/face_features/feature",
        help="Path to save face features.",
    )
    opt = parser.parse_args()

    # Run the main function
    add_persons(**vars(opt))
