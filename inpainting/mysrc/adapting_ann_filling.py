import json
import os
from PIL import Image
from tqdm import tqdm

images_dir = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/overlayed_imgs_controlnet_hed_blankImgs_filled"  # Path to the resized images
dataset_json = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_plus_withClassBalance_added_hed_blank.json"  # Original COCO JSON file
output_json = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_plus_withClassBalance_added_hed_blank_filled.json"  # Path for the updated JSON file

with open(dataset_json, 'r') as f:
    coco_data = json.load(f)


for image in tqdm(coco_data['images'],desc="Processing images"):
    if image['file_name'].startswith("blank"):
        resized_image_name = f"{image['file_name'].rsplit('.', 1)[0]}_filled.png"
        image_path = os.path.join(images_dir, resized_image_name)

        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                new_width, new_height = img.size

            old_width, old_height = image['width'], image['height']
            width_scale = new_width / old_width
            height_scale = new_height / old_height

            image['width'] = new_width
            image['height'] = new_height
            image['file_name'] = resized_image_name

            image_id = image['id']
            for annotation in coco_data['annotations']:
                if annotation['image_id'] == image_id:
                    old_bbox = annotation['bbox']
                    new_bbox = [
                        old_bbox[0] * width_scale,  # x
                        old_bbox[1] * height_scale,  # y
                        old_bbox[2] * width_scale,  # width
                        old_bbox[3] * height_scale   # height
                    ]
                    annotation['bbox'] = new_bbox

with open(output_json, 'w') as f:
    json.dump(coco_data, f, indent=4)

print(f"Updated COCO annotations saved to {output_json}")