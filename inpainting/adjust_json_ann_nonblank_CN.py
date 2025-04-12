import json
import cv2
from tqdm import tqdm

def adjust_coco_annotations(coco_path, image_dir, output_path):
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)
    
    for image in tqdm(coco_data['images'], desc='Adjusting annotations'):
        img_name = image['file_name']
        if img_name.startswith("blank"):
            continue
        expected_width, expected_height = image['width'], image['height']
        
        img_path = f"{image_dir}/{img_name}"
        actual_img = cv2.imread(img_path)
        if actual_img is None:
            print(f"Warning: Could not load image {img_name}")
            continue

        actual_height, actual_width = actual_img.shape[:2]
        
        width_scale = actual_width / expected_width
        height_scale = actual_height / expected_height

        image['width'] = actual_width
        image['height'] = actual_height

        for ann in coco_data['annotations']:
            if ann['image_id'] == image['id']:
                x, y, w, h = ann['bbox']
                ann['bbox'] = [
                    x * width_scale,
                    y * height_scale,
                    w * width_scale,
                    h * height_scale
                ]

    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"Updated annotations saved to {output_path}")

coco_annotation_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/synthesized_novel_DA_CB_combined_annotations.json'
image_directory = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/synthesized_novel_DA_CB_combined'
output_annotation_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/synthesized_novel_DA_CB_combined_annotations_adjusted.json'

adjust_coco_annotations(coco_annotation_path, image_directory, output_annotation_path)
