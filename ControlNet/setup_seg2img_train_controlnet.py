import os
import json
from PIL import Image
import cv2 
import numpy as np
from annotator.uniformer import UniformerDetector
from annotator.util import resize_image, HWC3
from tqdm import tqdm


source_img_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/bbox_images_train_all'
source_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/seg2img_controlnet_train_all_data/source'
target_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/seg2img_controlnet_train_all_data/target'
output_json = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/seg2img_controlnet_train_all_data/seg2img_odor_train_all_data.jsonl'


apply_uniformer = UniformerDetector()
os.makedirs(source_dir, exist_ok=True)
os.makedirs(target_dir, exist_ok=True)

json_data = []

for filename in tqdm(os.listdir(source_img_dir), desc="Processing Images"):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(source_img_dir, filename)
        image = cv2.imread(image_path)

        img = resize_image(HWC3(image), resolution = 512)
        cv2.imwrite(os.path.join(target_dir, filename), img)
        H, W, C = img.shape

        # apply uniformer for segmentation
        detected_map = apply_uniformer(resize_image(image, resolution=512))
        detected_map = HWC3(detected_map)
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        seg_filename = f"{os.path.splitext(filename)[0]}_seg.png"
        cv2.imwrite(os.path.join(source_dir, seg_filename), detected_map)

        class_name = filename.split('_')[-3]
        prompt = f"oil painting of {class_name} on canvas"

        json_line = {
            "source": f"{source_dir}/{seg_filename}",
            "target": f"{target_dir}/{filename}",
            "prompt": prompt
        }
        
        json_data.append(json_line)

with open(output_json, 'w') as f:
    for line in json_data:
        f.write(json.dumps(line) + '\n')

print(f"Processing complete. Data saved to {output_json}")