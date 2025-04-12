import os
import json
import cv2
from pycocotools.coco import COCO
from tqdm import tqdm

dir1 = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images_train_train_modified'
dir2 = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/controlnet_output_bbox_images_afterfinetuning15_b64_classimbalance_241124'
output_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/overlayed_imgs_controlnet_fe15_b64_s64' 
ann_file = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json"

coco = COCO(ann_file)
os.makedirs(output_dir, exist_ok=True)

filename_to_id = {img['file_name']: img['id'] for img in coco.dataset['images']}

def overlay_bbox_image(original_img, bbox_img, bbox_coords):
    x, y, w, h = map(int, bbox_coords)
    bbox_img_resized = cv2.resize(bbox_img, (w, h))
    original_img[y:y+h, x:x+w] = bbox_img_resized
    return original_img

def sanitize_class_name(class_name):
    return class_name.replace('/', '_').replace('\\', '_')

for original_img_filename in tqdm(os.listdir(dir1), desc="Overlaying images"):
    if original_img_filename.endswith('.jpg'):
        
        original_img_path = os.path.join(dir1, original_img_filename)
        original_img = cv2.imread(original_img_path)

        img_id = filename_to_id.get(original_img_filename)
        if img_id is None:
            print(f"No image ID found for {original_img_filename}")
            continue

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        annotations = coco.loadAnns(ann_ids)

        if not annotations:
            print(f"No annotations found for {original_img_filename}")
            continue

        for ann in annotations:
            bbox = ann['bbox']
            category_id = ann['category_id']
            class_name = sanitize_class_name(coco.loadCats([category_id])[0]['name'])
            ann_id = ann['id']

            bbox_img_filename = f"{original_img_filename.rsplit('.', 1)[0]}_{class_name}_{ann_id}_bbox_synthesized_1.png"
            bbox_img_path = os.path.join(dir2, bbox_img_filename)

            if os.path.exists(bbox_img_path):
                bbox_img = cv2.imread(bbox_img_path)
                original_img = overlay_bbox_image(original_img, bbox_img, bbox)
            else:
                print(f"bbox image {bbox_img_filename} does not exist!")

        output_img_path = os.path.join(output_dir, original_img_filename)
        cv2.imwrite(output_img_path, original_img)
