import os
import json
import cv2
import random
from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np

dir1 = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images_train_train_modified'
print(f"Original full images dir: {dir1}")
dir2 = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/FINAL_FINAL_flip_controlnet_hed_bbox_images_afterfinetuning20_b16_sdm256_classimbalance_0901'
print(f"BBox images dir: {dir2}")
# output_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/FINAL_FINAL_overlayed_imgs_controlnet_hed_f20_smin256max512_withClassBalance_random_overlaying_blank' 

output_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/overlayed_images_test_no_CB_no_blank' 
print(f"Output dir: {output_dir}")
ann_file = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json"
# output_ann_file = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/FINAL_FINAL_my_modified_train_plus_withClassBalance_added_hed_blank.json"
output_ann_file = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/dummy.json"

coco = COCO(ann_file)
os.makedirs(output_dir, exist_ok=True)

filename_to_id = {img['file_name']: img['id'] for img in coco.dataset['images']}

with open(ann_file, 'r') as f:
    coco_data = json.load(f)

annotation_id = max(ann['id'] for ann in coco_data['annotations']) + 1  

def overlay_bbox_image_noSmoothing(original_img, bbox_img, bbox_coords):
    x, y, w, h = map(int, bbox_coords)
    bbox_img_resized = cv2.resize(bbox_img, (w, h))
    original_img[y:y+h, x:x+w] = bbox_img_resized
    return original_img
    

def overlay_bbox_image(original_img, bbox_img, bbox_coords, feather_size=10, blur_strength=21):
    """
    feather_size (int): Width of the gradient transition zone at the edges.
    blur_strength (int): Kernel size for Gaussian blur to increase smoothness.
    """
    x, y, w, h = map(int, bbox_coords)
    bbox_img_resized = cv2.resize(bbox_img, (w, h))
    

    gradient_mask = np.zeros((h, w), dtype=np.float32)

    #  gradient transition zone
    for i in range(h):
        for j in range(w):
            dist_to_left = j
            dist_to_right = w - j - 1
            dist_to_top = i
            dist_to_bottom = h - i - 1
            min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
            
            gradient_mask[i, j] = np.clip(min_dist / feather_size, 0, 1)

    gradient_mask = cv2.GaussianBlur(gradient_mask, (blur_strength, blur_strength), 0)

    gradient_mask_3d = np.dstack([gradient_mask] * 3)

    roi = original_img[y:y+h, x:x+w].astype(np.float32)

    bbox_img_resized = bbox_img_resized.astype(np.float32)
    blended = (roi * (1 - gradient_mask_3d) + bbox_img_resized * gradient_mask_3d).astype(np.uint8)

    original_img[y:y+h, x:x+w] = blended

    return original_img


def sanitize_class_name(class_name):
    return class_name.replace('/', '_').replace('\\', '_')

def bbox_overlap(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

updated_images = []
updated_annotations = []

# step 1: process and overlay images ending with '_1'
existing_bboxes = {}
for original_img_filename in tqdm(os.listdir(dir1), desc="Overlaying _1 images"):
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

        existing_bboxes[original_img_filename] = [ann['bbox'] for ann in annotations]

        for ann in annotations:
            bbox = ann['bbox']  # COCO format: [x, y, width, height]
            category_id = ann['category_id']
            class_name = sanitize_class_name(coco.loadCats([category_id])[0]['name'])
            ann_id = ann['id']

            if class_name =="reptile/amphibia":
                class_name = "reptile_amphibia"

            bbox_img_filename = f"{original_img_filename.rsplit('.', 1)[0]}_{class_name}_{ann_id}_bbox_synthesized_1.png"
            bbox_img_path = os.path.join(dir2, bbox_img_filename)
            # print(f"bbox_img_path = {bbox_img_path}")

            if os.path.exists(bbox_img_path):
                bbox_img = cv2.imread(bbox_img_path)
                original_img = overlay_bbox_image(original_img, bbox_img, bbox)
            else:
                print(f"bbox image {bbox_img_filename} does not exist!")

        output_img_path = os.path.join(output_dir, original_img_filename)
        cv2.imwrite(output_img_path, original_img)

        # Update images and annotations for new JSON
        updated_images.append({
            "id": img_id,
            "file_name": original_img_filename,
            "width": original_img.shape[1],
            "height": original_img.shape[0]
        })

        for ann in annotations:
            updated_annotations.append(ann)


new_img_id = max(img['id'] for img in coco_data['images'])

# # step 2: randomly overlay images ending with '_2+' and save annotations
# for bbox_img_filename in tqdm(os.listdir(dir2), desc="Overlaying _2+ images"):
#     if bbox_img_filename.endswith('.png') and not bbox_img_filename.endswith('_1.png'):
        

#         parts = bbox_img_filename.split('_')
#         original_img_name = '_'.join(parts[:-5]) + '.jpg'
#         if original_img_name.endswith("reptile.jpg"):
#             original_img_name = original_img_name.replace("_reptile", "")

#         class_name = parts[-5]
#         if class_name =="amphibia":
#                 class_name = "reptile_amphibia"


#         ann_id = int(parts[-4]) 

#         ann_ids = coco.getAnnIds(imgIds=[filename_to_id[original_img_name]])
#         annotations = coco.loadAnns(ann_ids)
#         matching_ann = next((ann for ann in annotations if ann['id'] == ann_id), None)

#         if not matching_ann:
#             print(f"Matching annotation not found for {bbox_img_filename}")
#             continue

#         bbox_dims = matching_ann['bbox']  # use annotation bounding box dimensions
#         x, y, w, h = map(int, bbox_dims)

#         target_img_filename = random.choice(os.listdir(output_dir))
#         target_img_path = os.path.join(output_dir, target_img_filename)
#         target_img = cv2.imread(target_img_path)

        
        
#         img_h, img_w, _ = target_img.shape

#         if w > img_w or h > img_h:
#             print(f"BBox dimensions ({w}, {h}) exceed target image dimensions ({img_w}, {img_h}). Resizing to 256x256.")
#             w, h = 256, 256

#         if w > img_w or h > img_h:
#             print(f"BBox dimensions ({w}, {h}) still exceed target image dimensions ({img_w}, {img_h}). Resizing to 128x128.")
#             w, h = 128, 128

#         if w > img_w or h > img_h:
#             print(f"Even after resizing, bbox dimensions ({w}, {h}) exceed target image dimensions ({img_w}, {img_h}). Skipping.")
#             continue
        
#         bbox_img_path = os.path.join(dir2, bbox_img_filename)
#         bbox_img = cv2.imread(bbox_img_path)
#         bbox_img_resized = cv2.resize(bbox_img, (w, h))

        
#         for attempt in range(10):
#             try:
#                 x = random.randint(0, img_w - w)
#                 y = random.randint(0, img_h - h)
#                 new_bbox = [x, y, w, h]

#                 if all(not bbox_overlap(existing_bbox, new_bbox) for existing_bbox in existing_bboxes[target_img_filename]):
#                     break
#             except ValueError as e:
#                 print(f"ValueError: {e} for bbox_img_filename {bbox_img_filename}. Skipping this overlay.")
#                 continue
#         else:
#             # If all attempts fail, create a blank image
#             print(f"Failed to find a non-overlapping position for {bbox_img_filename}. Creating a blank image.")

#             # Define blank image dimensions
#             blank_img_h, blank_img_w = 640, 640
#             blank_img = 255 * np.ones((blank_img_h, blank_img_w, 3), dtype=np.uint8)

#             new_img_filename = f"blank_image_{annotation_id}.png"
#             new_img_path = os.path.join(output_dir, new_img_filename)

            
#             # oversampled_img_filename = random.choice(os.listdir(output_dir))
#             # oversampled_img_path = os.path.join(output_dir, oversampled_img_filename)
#             # oversampled_img = cv2.imread(oversampled_img_path)
#             # ov_img_h, ov_img_w, _ = oversampled_img.shape
#             # new_img_filename = f"oversampled_image_{new_img_id}.png"
#             # new_img_path = os.path.join(output_dir, new_img_filename)

#             new_img_id = new_img_id + 1

            

#             existing_bboxes[new_img_filename] = []


#             updated_images.append({
#                 "id": new_img_id,
#                 "file_name": new_img_filename,
#                 "width": blank_img_w,
#                 "height": blank_img_h
#             })

#             filename_to_id[new_img_filename] = new_img_id

#             if w > blank_img_w or h > blank_img_h:
#                 print(f"BBox dimensions ({w}, {h}) exceed target image dimensions ({blank_img_w}, {blank_img_h}). Resizing to 256x256.")
#                 w, h = 256, 256

#             if w > blank_img_w or h > blank_img_h:
#                 print(f"BBox dimensions ({w}, {h}) still exceed target image dimensions ({blank_img_w}, {blank_img_h}). Resizing to 128x128.")
#                 w, h = 128, 128

#             if w > blank_img_w or h > blank_img_h:
#                 print(f"Even after resizing, bbox dimensions ({w}, {h}) exceed target image dimensions ({blank_img_w}, {blank_img_h}). Skipping.")
#                 continue
            

#             # Overlay the bbox onto the blank image
#             for _ in range(10):
#                 x = random.randint(0, blank_img_w - w)
#                 y = random.randint(0, blank_img_h - h)
#                 new_bbox = [x, y, w, h]
#                 if all(not bbox_overlap(existing_bbox, new_bbox) for existing_bbox in existing_bboxes[new_img_filename]):
#                     break

#             target_img = blank_img
#             target_img = overlay_bbox_image(target_img, bbox_img_resized, new_bbox)
#             existing_bboxes[new_img_filename].append(new_bbox)
#             updated_annotations.append({
#                 "id": annotation_id,
#                 "image_id": new_img_id,
#                 "category_id": matching_ann['category_id'],
#                 "bbox": new_bbox,
#                 "area": w * h,
#                 "iscrowd": 0
#             })
#             annotation_id += 1

#             # Save blank image with overlayed bbox
#             cv2.imwrite(new_img_path, target_img)
#             continue

#         # Overlay onto the original target image
#         target_img = overlay_bbox_image(target_img, bbox_img_resized, new_bbox)

#         # Update annotations and existing bounding boxes
#         existing_bboxes.setdefault(target_img_filename, []).append(new_bbox)
#         updated_annotations.append({
#             "id": annotation_id,
#             "image_id": filename_to_id[target_img_filename],
#             "category_id": matching_ann['category_id'],
#             "bbox": new_bbox,
#             "area": w * h,
#             "iscrowd": 0
#         })
#         annotation_id += 1

#         # Save updated target image
#         cv2.imwrite(target_img_path, target_img)

updated_coco_data = {
    "images": updated_images,
    "annotations": updated_annotations,
    "categories": coco_data['categories']
}

with open(output_ann_file, 'w') as f:
    json.dump(updated_coco_data, f)

print(f"Overlaying complete. Updated COCO JSON saved to {output_ann_file}")