import json
import numpy as np

def calculate_bbox_and_image_stats(coco_json_path):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    bboxes = [annotation['bbox'] for annotation in coco_data['annotations']]
    widths = [bbox[2] for bbox in bboxes]
    heights = [bbox[3] for bbox in bboxes]

    bbox_stats = {
        'average_width': np.mean(widths),
        'median_width': np.median(widths),
        'max_width': np.max(widths),
        'min_width': np.min(widths),
        'average_height': np.mean(heights),
        'median_height': np.median(heights),
        'max_height': np.max(heights),
        'min_height': np.min(heights)
    }

    img_widths = [image['width'] for image in coco_data['images']]
    img_heights = [image['height'] for image in coco_data['images']]

    image_stats = {
        'average_img_width': np.mean(img_widths),
        'median_img_width': np.median(img_widths),
        'max_img_width': np.max(img_widths),
        'min_img_width': np.min(img_widths),
        'average_img_height': np.mean(img_heights),
        'median_img_height': np.median(img_heights),
        'max_img_height': np.max(img_heights),
        'min_img_height': np.min(img_heights)
    }

    return bbox_stats, image_stats

coco_json_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json'
bbox_stats, image_stats = calculate_bbox_and_image_stats(coco_json_path)

print("Bounding Box Statistics:")
print(f"Average Width: {bbox_stats['average_width']}")
print(f"Median Width: {bbox_stats['median_width']}")
print(f"Max Width: {bbox_stats['max_width']}")
print(f"Min Width: {bbox_stats['min_width']}")
print(f"Average Height: {bbox_stats['average_height']}")
print(f"Median Height: {bbox_stats['median_height']}")
print(f"Max Height: {bbox_stats['max_height']}")
print(f"Min Height: {bbox_stats['min_height']}")

print("\nImage Statistics:")
print(f"Average Image Width: {image_stats['average_img_width']}")
print(f"Median Image Width: {image_stats['median_img_width']}")
print(f"Max Image Width: {image_stats['max_img_width']}")
print(f"Min Image Width: {image_stats['min_img_width']}")
print(f"Average Image Height: {image_stats['average_img_height']}")
print(f"Median Image Height: {image_stats['median_img_height']}")
print(f"Max Image Height: {image_stats['max_img_height']}")
print(f"Min Image Height: {image_stats['min_img_height']}")
