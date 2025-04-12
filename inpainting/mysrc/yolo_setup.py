import os
import json
import shutil

json_file_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolo/annotations/train_annotations.json'
source_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/' 
target_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolo/images/train/' 


if not os.path.exists(target_dir):
    os.makedirs(target_dir)

with open(json_file_path, 'r') as f:
    data = json.load(f)

image_files = [image['file_name'] for image in data['images']]

for image_file in image_files:
    src_file = os.path.join(source_dir, image_file)
    dest_file = os.path.join(target_dir, image_file)
    
    if os.path.exists(src_file):
        shutil.copy(src_file, dest_file)
        print(f"Copied: {image_file}")
    else:
        print(f"File not found: {image_file}")

print("Finished copying files.")
