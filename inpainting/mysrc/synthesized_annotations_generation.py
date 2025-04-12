import json
import os
from collections import defaultdict
import re
from tqdm import tqdm
from PIL import Image

def load_coco_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_coco_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def parse_synthesized_image_name(file_name):
    parts = file_name.split('_')
    annotation_id = parts[-2]
    category_name = parts[-3]
    original_image_name = '_'.join(parts[:-3])
    return original_image_name, category_name, annotation_id

def adjust_bbox(bbox, original_size, new_size):
    orig_w, orig_h = original_size
    new_w, new_h = new_size
    x, y, width, height = bbox
    scale_w = new_w / orig_w
    scale_h = new_h / orig_h
    return [x * scale_w, y * scale_h, width * scale_w, height * scale_h]

def resize_image(image_path, output_path, size):
    with Image.open(image_path) as img:
        resized_img = img.resize(size)
        resized_img.save(output_path)

def generate_synthesized_coco_json(original_coco, synthesized_images_path, output_json_path): #  resized_images_path="/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/synthesized_images_a100"
    new_coco = {
        "images": [],
        "annotations": [],
        "categories": original_coco["categories"]
    }

    annotation_by_id = {ann["id"]: ann for ann in original_coco["annotations"]}
    image_by_id = {img["id"]: img for img in original_coco["images"]}

    annotation_id_counter = max(annotation_by_id.keys()) + 1
    image_id_counter = max(image_by_id.keys()) + 1

    synthesized_images = [f for f in os.listdir(synthesized_images_path) if f.endswith('.png')]

    for synth_img in tqdm(synthesized_images,desc="Processing the synthesized annotations"):
        original_image_name, category_name, annotation_id = parse_synthesized_image_name(synth_img)
        if annotation_id is None:
            print(f"Skipping synthesized image with invalid name: {synth_img}")
            continue
        
        annotation_id = int(annotation_id)
        

        original_image = next((img for img in original_coco["images"] if os.path.splitext(img["file_name"])[0] == original_image_name), None)
        if not original_image:
            print(f"Original image not found for synthesized image: {synth_img}")
            continue

        original_size = (original_image["width"], original_image["height"])
        
        synthesized_image_path = os.path.join(synthesized_images_path, synth_img)

        # resized_image_path = os.path.join(resized_images_path, synth_img)
        # resize_image(synthesized_image_path, resized_image_path, original_size)
        with Image.open(synthesized_image_path) as img:
            new_size = img.size

        original_annotation = annotation_by_id.get(annotation_id)
        if not original_annotation:
            print(f"Original annotation not found for synthesized image: {synth_img}")
            continue

        new_image = {
            "id": image_id_counter,
            "file_name": synth_img,
            "height": new_size[1], # "height": original_size[1],
            "width": new_size[0]  # "width": original_size[0]
        }
        new_coco["images"].append(new_image)

        # Adjust the bounding box
        adjusted_bbox = adjust_bbox(original_annotation["bbox"], original_size, new_size)

        # Create a new annotation entry
        new_annotation = {
            "id": annotation_id_counter,
            "image_id": image_id_counter,
            "category_id": original_annotation["category_id"],
            "bbox": adjusted_bbox, # original_annotation["bbox"],
            "area": adjusted_bbox[2] * adjusted_bbox[3], # original_annotation["area"]
            "iscrowd": original_annotation["iscrowd"],
            "rotation": original_annotation.get("rotation", 0),
            "supercategory": original_annotation.get("supercategory", ""),
            "occluded": original_annotation.get("occluded", False),
            "label_id": original_annotation.get("label_id", "")
        }
        new_coco["annotations"].append(new_annotation)

        image_id_counter += 1
        annotation_id_counter += 1

    save_coco_json(new_coco, output_json_path)

original_coco_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/instances_train.json'
synthesized_images_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/synthesized_images_a100'
output_json_path = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/synthesized_ann_train.json'

original_coco = load_coco_json(original_coco_path)
generate_synthesized_coco_json(original_coco, synthesized_images_path, output_json_path)
