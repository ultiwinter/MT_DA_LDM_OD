import json
import os
from PIL import Image
from tqdm import tqdm


original_coco_path = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json"
augmented_images_dir = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/synthesized_borderobj_DA_images"
output_coco_path = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/synthesized_borderobj_annotations.json"


with open(original_coco_path, 'r') as f:
    coco_data = json.load(f)


new_coco_data = coco_data.copy()


image_id_map = {}
new_images = []
new_annotations = []

for image in tqdm(coco_data['images'], desc="Processing Images"):
    orig_image_name = image['file_name']

    augmented_image_name = f"{orig_image_name.rsplit('.', 1)[0]}_synthesized.png"
    augmented_image_path = os.path.join(augmented_images_dir, augmented_image_name)

    try:
        with Image.open(augmented_image_path) as img:
            new_width, new_height = img.size
    except FileNotFoundError:
        print(f"Augmented image not found: {augmented_image_name}")
        continue

    new_image = {
        "id": len(new_images) + 1,
        "file_name": augmented_image_name,
        "width": new_width,
        "height": new_height,
    }
    image_id_map[image['id']] = new_image['id']
    new_images.append(new_image)

    for annotation in coco_data['annotations']:
        if annotation['image_id'] == image['id']:
            orig_width, orig_height = image['width'], image['height']
            scale_x = new_width / orig_width
            scale_y = new_height / orig_height

            x, y, w, h = annotation['bbox']
            new_bbox = [x * scale_x, y * scale_y, w * scale_x, h * scale_y]

            new_annotation = annotation.copy()
            new_annotation['image_id'] = new_image['id']
            new_annotation['bbox'] = new_bbox
            new_annotations.append(new_annotation)

new_coco_data['images'] = new_images
new_coco_data['annotations'] = new_annotations

with open(output_coco_path, 'w') as f:
    json.dump(new_coco_data, f, indent=4)

print(f"Updated COCO JSON saved to {output_coco_path}")