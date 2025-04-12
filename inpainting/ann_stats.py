import json

coco_json_path = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json"
coco_json_path = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/instances_train.json"

with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

num_images = len(coco_data.get('images', []))

num_bboxes = len(coco_data.get('annotations', []))

print(f"Number of images: {num_images}")
print(f"Number of bounding boxes: {num_bboxes}")
