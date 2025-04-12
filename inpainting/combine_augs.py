import os
import cv2
import json
from tqdm import tqdm
import numpy as np

def process_images_with_augmented_bboxes(
    orig_img_dir, augmented_img_dir, coco_json_path, output_dir, output_json_path
):
    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    images = {img["id"]: img for img in coco_data["images"]}

    annotations = coco_data["annotations"]


    annotations_by_image = {}
    for annotation in annotations:
        image_id = annotation["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    os.makedirs(output_dir, exist_ok=True)

    
    new_annotations = []
    new_images = []

    for image_id, image_annotations in tqdm(
        annotations_by_image.items(), desc="Processing Images"
    ):
        if image_id not in images:
            print(f"Image ID {image_id} not found in COCO data")
            continue

        image_info = images[image_id]
        orig_img_name = image_info["file_name"]
        orig_img_path = os.path.join(orig_img_dir, orig_img_name)

        original_img = cv2.imread(orig_img_path)
        if original_img is None:
            print(f"Failed to load original image: {orig_img_path}")
            continue

        orig_h, orig_w = original_img.shape[:2]

        output_img = None
        updated_image_info = image_info.copy()

        for annotation in image_annotations:
            bbox = annotation["bbox"] 
            ann_id = annotation["id"]
            category_id = annotation["category_id"]
            class_name = coco_data["categories"][category_id]["name"]
            if class_name == "reptile/amphibia":
                class_name = "reptile-amphibia"

            orig_img_base = orig_img_name.rsplit(".", 1)[0]  # rsplit for robustness
            aug_img_name = f"{orig_img_base}_{class_name}_{ann_id}_mask.png"
            aug_img_path = os.path.join(augmented_img_dir, aug_img_name)

            aug_img = cv2.imread(aug_img_path)
            if aug_img is None:
                print(f"Failed to load augmented image: {aug_img_path}")
                continue

            aug_h, aug_w = aug_img.shape[:2]

            if output_img is None:
                # output_img = cv2.resize(original_img, (aug_w, aug_h))
                output_img = np.zeros((aug_h, aug_w, 3), dtype=np.uint8) # for combining masks
                scale_w = aug_w / orig_w
                scale_h = aug_h / orig_h
                updated_image_info["width"] = aug_w
                updated_image_info["height"] = aug_h

            x, y, w, h = bbox
            rescaled_bbox = [
                int(x * scale_w),
                int(y * scale_h),
                int(w * scale_w),
                int(h * scale_h),
            ]

            x, y, w, h = rescaled_bbox
            aug_region = aug_img[y : y + h, x : x + w]

            try:
                output_img[y : y + h, x : x + w] = aug_region
                pass
            except Exception as e:
                print(f"Error overlaying bbox on {orig_img_name}: {e}")
                continue

            updated_annotation = annotation.copy()
            updated_annotation["bbox"] = [
                rescaled_bbox[0],
                rescaled_bbox[1],
                rescaled_bbox[2],
                rescaled_bbox[3],
            ]
            new_annotations.append(updated_annotation)

        if output_img is not None:
            output_img_name = f"{orig_img_base}_combined_augmented.png"
            output_img_path = os.path.join(output_dir, output_img_name)
            cv2.imwrite(output_img_path, output_img)
        else:
            print(f"Skipping {orig_img_name} as no output was generated.")

        new_images.append(updated_image_info)
    
    # save the updated COCO JSON
    updated_coco_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco_data["categories"],
    }

    with open(output_json_path, "w") as f:
        json.dump(updated_coco_data, f, indent=4)

    print("Processing complete. Updated JSON saved to:", output_json_path)


    


orig_img_dir = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images"
augmented_img_dir = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/whole_masks" # synthesized_gradient_DA_images synthesized_random_partial_RWML_DA
coco_json_path = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json"
output_dir = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/combined_whole_masks" # synthesized_random_partial_DA_combined
output_json_path = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/combined_whole_masks.json" # synthesized_random_partial_DA_combined_annotations.json


process_images_with_augmented_bboxes(orig_img_dir, augmented_img_dir, coco_json_path, output_dir, output_json_path)
