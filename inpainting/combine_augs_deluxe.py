import os
import cv2
import json
from tqdm import tqdm

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

    processed_images = set()
    new_image_id = max(img["id"] for img in coco_data["images"]) + 1
    new_annotation_id = max(ann["id"] for ann in coco_data["annotations"]) + 1

    for idx in range(1, 11):  # Iterate through indices _1 to _10
        for image_id, image_annotations in tqdm(
            annotations_by_image.items(), desc=f"Processing Index {idx}"
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

            combined_img = None
            has_augmentations = False

            updated_annotations = []

            for ann_idx, annotation in enumerate(image_annotations):
                bbox = annotation["bbox"].copy()
                ann_id = annotation["id"]
                category_id = annotation["category_id"]
                class_name = coco_data["categories"][category_id]["name"]

                if class_name == "reptile/amphibia":
                    class_name = "reptile-amphibia"

                orig_img_base = os.path.splitext(orig_img_name)[0]
                aug_img_name = f"{orig_img_base}_{class_name}_{ann_id}_synthesized_{idx}.png"
                aug_img_path = os.path.join(augmented_img_dir, aug_img_name)
                if os.path.exists(aug_img_path):
                    aug_img = cv2.imread(aug_img_path)
                    if aug_img is None:
                        print(f"Failed to load augmented image: {aug_img_path}")
                        continue

                    aug_h, aug_w = aug_img.shape[:2]

                    if combined_img is None:
                        combined_img = cv2.resize(original_img, (aug_w, aug_h))

                    # orig_h, orig_w = combined_img.shape[:2]

                    scale_w = aug_w / orig_w
                    scale_h = aug_h / orig_h
                    ##################################
                    x, y, w, h = bbox
                    rescaled_bbox = [
                        int(x * scale_w),
                        int(y * scale_h),
                        int(w * scale_w),
                        int(h * scale_h),
                    ]
                    #################################
                    # rescaled_bbox = [
                    #     max(0, min(int(x * scale_w), aug_w - 1)),
                    #     max(0, min(int(y * scale_h), aug_h - 1)),
                    #     max(1, min(int(w * scale_w), aug_w - int(x * scale_w) - 1)),
                    #     max(1, min(int(h * scale_h), aug_h - int(y * scale_h) - 1)),
                    # ]

                    x, y, w, h = rescaled_bbox

                    # image_annotations[ann_idx]["bbox"] = rescaled_bbox
                    # annotation["bbox"] = rescaled_bbox

                    try:
                        aug_region = cv2.resize(aug_img[y : y + h, x : x + w], (w, h))
                        # aug_region = aug_img[y : y + h, x : x + w]
                        combined_img[y : y + h, x : x + w] = aug_region
                        has_augmentations = True
                    except Exception as e:
                        print(f"Error overlaying bbox on {orig_img_name}: {e}")


                    new_annotation = annotation.copy()
                    new_annotation["id"] = new_annotation_id
                    new_annotation["image_id"] = new_image_id
                    new_annotation["bbox"] = rescaled_bbox
                    updated_annotations.append(new_annotation)
                    new_annotation_id += 1

            if has_augmentations and combined_img is not None:
                combined_img_name = f"{orig_img_base}_combined_{idx}.png"
                combined_img_path = os.path.join(output_dir, combined_img_name)
                cv2.imwrite(combined_img_path, combined_img)

                new_image_info = image_info.copy()
                new_image_info["id"] = new_image_id
                new_image_info["file_name"] = combined_img_name
                new_images.append(new_image_info)

                new_image_info["width"] = combined_img.shape[1]  # width
                new_image_info["height"] = combined_img.shape[0]  # height

                # # Update annotations with new unique IDs and new image ID
                # for annotation in image_annotations:
                #     new_annotation = annotation.copy()
                #     new_annotation["id"] = new_annotation_id
                #     new_annotation["image_id"] = new_image_id
                #     new_annotations.append(new_annotation)
                #     new_annotation_id += 1

                new_annotations.extend(updated_annotations)

                new_image_id += 1
                processed_images.add(orig_img_name)

    for image_info in coco_data["images"]:
        orig_img_name = image_info["file_name"]
        if orig_img_name not in processed_images:
            orig_img_path = os.path.join(orig_img_dir, orig_img_name)
            if os.path.exists(orig_img_path):
                output_img_path = os.path.join(output_dir, orig_img_name)
                cv2.imwrite(output_img_path, cv2.imread(orig_img_path))

            image_id = image_info["id"]
            new_image_info = image_info.copy()
            new_image_info["id"] = new_image_id
            new_images.append(new_image_info)

            if image_id in annotations_by_image:
                for annotation in annotations_by_image[image_id]:
                    new_annotation = annotation.copy()
                    new_annotation["id"] = new_annotation_id
                    new_annotation["image_id"] = new_image_id
                    new_annotations.append(new_annotation)
                    new_annotation_id += 1

            new_image_id += 1

    updated_coco_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco_data["categories"],
    }

    with open(output_json_path, "w") as f:
        json.dump(updated_coco_data, f, indent=4)

    print("Processing complete. Updated JSON saved to:", output_json_path)


orig_img_dir = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images_train_train_modified"
augmented_img_dir = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/synthesized_random_partial_RWML_DA_CB"
coco_json_path = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json"
output_dir = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/TEST8_combined_random_partial"
output_json_path = "/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/TEST8_combined_random_partial_annotations.json"
print(augmented_img_dir)
process_images_with_augmented_bboxes(orig_img_dir, augmented_img_dir, coco_json_path, output_dir, output_json_path)
