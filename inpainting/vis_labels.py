import json
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

images_dir = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images" 
json_path = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/instances_all.json"  
output_dir = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/vis_labels_orig_ODOR" 
os.makedirs(output_dir, exist_ok=True)

with open(json_path, "r") as f:
    data = json.load(f)

image_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

category_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

def draw_and_save_bboxes(image_path, annotations, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for ann in annotations:
        x_min, y_min, width, height = ann["bbox"]
        x_max, y_max = x_min + width, y_min + height
        class_name = category_id_to_name[ann["category_id"]]

        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        cv2.putText(img, class_name, (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # convert back to BGR for saving
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_bgr)

print("Visualizing bbox labels...")
for image_id, file_name in tqdm(image_id_to_file.items()):
    
    image_path = os.path.join(images_dir, file_name)

    image_annotations = [ann for ann in data["annotations"] if ann["image_id"] == image_id]

    if image_annotations:
        # print(f"Processing {file_name} with {len(image_annotations)} annotations...")
        output_path = os.path.join(output_dir, file_name)
        draw_and_save_bboxes(image_path, image_annotations, output_path)
    else:
        print(f"No annotations found for {file_name}.")

print(f"Annotated images saved to {output_dir}")
