import json
import os
import shutil
from tqdm import tqdm

def copy_images_from_coco_annotations(coco_json_path, src_dir, target_dir):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    os.makedirs(target_dir, exist_ok=True)
    
    image_filenames = {img['file_name'] for img in coco_data.get('images', [])}
    
    for image_name in tqdm(image_filenames, desc="Copying images", unit="file"):
        src_path = os.path.join(src_dir, image_name)
        target_path = os.path.join(target_dir, image_name)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, target_path)
        else:
            print(f"Image not found: {src_path}")

    print("Image copying completed!")

coco_json_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json'
src_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images'
target_dir = '//home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images_train_train_modified' 

copy_images_from_coco_annotations(coco_json_path, src_dir, target_dir)
