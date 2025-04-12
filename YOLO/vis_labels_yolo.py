import os
import cv2
from tqdm import tqdm

def visualize_yolo_labels(images_dir, labels_dir, output_dir, class_names=None):
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png')

    for img_file in tqdm(os.listdir(images_dir)):
        if not img_file.lower().endswith(image_extensions):
            continue

        image_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')

        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Couldn't read image: {image_path}")
            continue

        height, width, _ = image.shape

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id, x_center, y_center, w, h = map(float, parts)
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height

                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)

                    label = f"{int(class_id)}" if not class_names else class_names[int(class_id)]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, image)


images_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_novel/baseline_novel/images/train'
labels_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_novel/baseline_novel/labels/train'
output_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/vis_labels_yolo_baseline_novel'

visualize_yolo_labels(images_dir, labels_dir, output_dir)
