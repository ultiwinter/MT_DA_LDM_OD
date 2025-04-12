import os
import json
from PIL import Image
from tqdm import tqdm
coco_annotation_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/instances_train.json'
image_src_folder = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images'
output_image_folder = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/bbox_images_train_all'
images_txt_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/train_images.txt'
labels_txt_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/train_labels.txt'

os.makedirs(output_image_folder, exist_ok=True)

def sanitize_class_name(class_name):
    return class_name.replace('/', '_').replace('\\', '_')

image_paths = []
labels = []

with open(coco_annotation_path, 'r') as f:
    coco_data = json.load(f)

category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

for annotation in tqdm(coco_data['annotations'],desc="Processing annotations"):
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    bbox = annotation['bbox']
    ann_id = annotation['id']

    image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
    image_name = image_info['file_name']
    image_path = os.path.join(image_src_folder, image_name)
    
    with Image.open(image_path) as img:
        x, y, width, height = bbox
        cropped_img = img.crop((x, y, x + width, y + height))
        
        class_name = sanitize_class_name(category_id_to_name.get(category_id, str(category_id)))
        
        output_file_name = f"{image_name.rsplit('.', 1)[0]}_{class_name}_{ann_id}_bbox.png"
        output_file_path = os.path.join(output_image_folder, output_file_name)
        
        cropped_img.save(output_file_path)
        
        relative_output_path = os.path.relpath(output_file_path, output_image_folder)
        image_paths.append(relative_output_path)
        labels.append(class_name)

with open(images_txt_path, 'w') as f:
    for path in image_paths:
        f.write(f"{path}\n")

with open(labels_txt_path, 'w') as f:
    for label in labels:
        f.write(f"{label}\n")


with open(images_txt_path, 'r') as img_file, open(labels_txt_path, 'r') as lbl_file:
    img_lines = img_file.readlines()
    lbl_lines = lbl_file.readlines()

    assert len(img_lines) == len(lbl_lines), "The number of entries in images.txt and labels.txt must match."

    for i, (img_line, lbl_line) in tqdm(enumerate(zip(img_lines, lbl_lines)),desc="Validating the consistency of image.txt and labels.txt"):
        img_line = img_line.strip()
        lbl_line = lbl_line.strip()
        if not lbl_line in img_line:
            raise ValueError(f"Mismatch at line {i + 1}: image path '{img_line}' does not contain the expected label '{lbl_line}'.")


print("Processing completed successfully.")
