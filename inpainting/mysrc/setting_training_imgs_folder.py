import os
import shutil
import json
from tqdm import tqdm


def copy_images_from_coco_json(json_file, source_dir, target_dir):
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for image_info in tqdm(coco_data['images'],desc="Copying images"):
        image_file = image_info['file_name']
        
        source_image_path = os.path.join(source_dir, image_file)
        
        target_image_path = os.path.join(target_dir, image_file)
        
        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, target_image_path)
            print(f"Copied {image_file} to {target_dir}")
        else:
            print(f"Image {image_file} not found in source directory.")

json_file = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json'
source_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images'
target_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images_train_train_modified'

copy_images_from_coco_json(json_file, source_dir, target_dir)
