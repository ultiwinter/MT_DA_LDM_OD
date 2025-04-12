import os
import json
from shutil import copyfile
from tqdm import tqdm

train_coco_json_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json '
val_coco_json_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_val_split.json '
images_src_folder = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/'
output_folder = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_orig'

def process_coco_json(coco_json_path, img_type, category_mapping):
    os.makedirs(os.path.join(output_folder, f'{img_type}/images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, f'{img_type}/labels'), exist_ok=True)

    with open(coco_json_path) as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}

    for ann in tqdm(coco_data['annotations'], desc=f"Processing {img_type} annotations to YOLO format:"):
        img_id = ann['image_id']
        img_data = images[img_id]
        img_filename = img_data['file_name']

        img_output_path = os.path.join(output_folder, f'{img_type}/images', img_filename)
        if not os.path.exists(img_output_path):
            copyfile(os.path.join(images_src_folder, img_filename), img_output_path)

        label_output_path = os.path.join(output_folder, f'{img_type}/labels', os.path.splitext(img_filename)[0] + '.txt')
        with open(label_output_path, 'a') as label_file:
            bbox = ann['bbox']
            category_id = ann['category_id']
            yolo_category_id = category_mapping[category_id] 
            x_center = (bbox[0] + bbox[2] / 2) / img_data['width']
            y_center = (bbox[1] + bbox[3] / 2) / img_data['height']
            width = bbox[2] / img_data['width']
            height = bbox[3] / img_data['height']

            label_file.write(f"{yolo_category_id} {x_center} {y_center} {width} {height}\n")

def create_category_mapping(coco_json_path):
    with open(coco_json_path) as f:
        coco_data = json.load(f)
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    category_names = {idx: cat['name'] for idx, cat in enumerate(coco_data['categories'])}
    return categories, category_names

train_category_mapping, train_category_names = create_category_mapping(train_coco_json_path)

process_coco_json(train_coco_json_path, 'train', train_category_mapping)
process_coco_json(val_coco_json_path, 'val', train_category_mapping)


names_formatted = [f"{idx}: {name}" for idx, name in train_category_names.items()]
yaml_content = f"""
train: {os.path.join(output_folder, 'train/images')}
val: {os.path.join(output_folder, 'val/images')}

nc: {len(train_category_names)}
names: {names_formatted}
"""
with open(os.path.join(output_folder, 'data.yaml'), 'w') as yaml_file:
    yaml_file.write(yaml_content)

print("Dataset prepared successfully!")
