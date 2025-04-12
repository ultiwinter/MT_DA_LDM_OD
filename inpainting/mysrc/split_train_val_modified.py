import json
from collections import defaultdict
import random

with open('/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/instances_train.json', 'r') as f:
    coco_data = json.load(f)

images = coco_data['images']
annotations = coco_data['annotations']
categories = coco_data['categories']

image_annotations = defaultdict(list)
class_counts = defaultdict(int) 

for annotation in annotations:
    image_annotations[annotation['image_id']].append(annotation)
    class_counts[annotation['category_id']] += 1

random.shuffle(images)

train_images = []
val_images = []
train_class_counts = defaultdict(int)
val_class_counts = defaultdict(int)
train_annotations = []
val_annotations = []

target_train_size = int(len(images) * 0.8)

for image in images:
    image_id = image['id']
    image_annots = image_annotations[image_id]
    image_classes = {annot['category_id'] for annot in image_annots}

    if len(train_images) < target_train_size:
        train_images.append(image)
        train_annotations.extend(image_annots)
        for annot in image_annots:
            train_class_counts[annot['category_id']] += 1
    else:
        val_images.append(image)
        val_annotations.extend(image_annots)
        for annot in image_annots:
            val_class_counts[annot['category_id']] += 1

all_classes = {category['id'] for category in categories}
missing_classes_in_train = all_classes - set(train_class_counts.keys())
missing_classes_in_val = all_classes - set(val_class_counts.keys())

for class_id in missing_classes_in_train:
    for image in val_images:
        image_id = image['id']
        image_classes = {annot['category_id'] for annot in image_annotations[image_id]}
        if class_id in image_classes:
            val_images.remove(image)
            train_images.append(image)
            image_annots = image_annotations[image_id]
            for annot in image_annots:
                train_class_counts[annot['category_id']] += 1
                val_class_counts[annot['category_id']] -= 1
            train_annotations.extend(image_annots)
            val_annotations = [annot for annot in val_annotations if annot not in image_annots]
            break

for class_id in missing_classes_in_val:
    for image in train_images:
        image_id = image['id']
        image_classes = {annot['category_id'] for annot in image_annotations[image_id]}
        if class_id in image_classes:
            train_images.remove(image)
            val_images.append(image)
            image_annots = image_annotations[image_id]
            for annot in image_annots:
                val_class_counts[annot['category_id']] += 1
                train_class_counts[annot['category_id']] -= 1
            val_annotations.extend(image_annots)
            train_annotations = [annot for annot in train_annotations if annot not in image_annots]
            break

def ensure_all_classes_present(train_class_counts, val_class_counts, all_classes):
    missing_classes_in_train = all_classes - set(train_class_counts.keys())
    missing_classes_in_val = all_classes - set(val_class_counts.keys())

    for class_id in missing_classes_in_train:
        for image in val_images:
            image_id = image['id']
            image_classes = {annot['category_id'] for annot in image_annotations[image_id]}
            if class_id in image_classes:
                val_images.remove(image)
                train_images.append(image)
                image_annots = image_annotations[image_id]
                for annot in image_annots:
                    train_class_counts[annot['category_id']] += 1
                    val_class_counts[annot['category_id']] -= 1
                train_annotations.extend(image_annots)
                val_annotations = [annot for annot in val_annotations if annot not in image_annots]
                break

    for class_id in missing_classes_in_val:
        for image in train_images:
            image_id = image['id']
            image_classes = {annot['category_id'] for annot in image_annotations[image_id]}
            if class_id in image_classes:
                train_images.remove(image)
                val_images.append(image)
                image_annots = image_annotations[image_id]
                for annot in image_annots:
                    val_class_counts[annot['category_id']] += 1
                    train_class_counts[annot['category_id']] -= 1
                val_annotations.extend(image_annots)
                train_annotations = [annot for annot in train_annotations if annot not in image_annots]
                break

ensure_all_classes_present(train_class_counts, val_class_counts, all_classes)

def check_no_overlap(train_images, val_images):
    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}
    overlap = train_image_ids.intersection(val_image_ids)
    if overlap:
        print(f"Error: Overlapping images detected. Overlapping image IDs: {overlap}")
    else:
        print("No overlapping images between train and validation splits.")

def check_all_annotations_utilized(train_annotations, val_annotations, original_annotations):
    train_val_annotation_ids = {annot['id'] for annot in train_annotations + val_annotations}
    original_annotation_ids = {annot['id'] for annot in original_annotations}
    unused_annotations = original_annotation_ids - train_val_annotation_ids
    if unused_annotations:
        print(f"Error: Not all annotations are utilized. Unused annotation IDs: {unused_annotations}")
    else:
        print("All annotations are utilized.")

# Check that all images are utilized
def check_all_images_utilized(train_images, val_images, original_images):
    train_val_image_ids = {img['id'] for img in train_images + val_images}
    original_image_ids = {img['id'] for img in original_images}
    unused_images = original_image_ids - train_val_image_ids
    if unused_images:
        print(f"Error: Not all images are utilized. Unused image IDs: {unused_images}")
    else:
        print("All images are utilized.")

check_no_overlap(train_images, val_images)
check_all_annotations_utilized(train_annotations, val_annotations, annotations)
check_all_images_utilized(train_images, val_images, images)

train_data = {
    'images': train_images,
    'annotations': train_annotations,
    'categories': categories
}

val_data = {
    'images': val_images,
    'annotations': val_annotations,
    'categories': categories
}

with open('/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json', 'w') as f:
    json.dump(train_data, f)

with open('/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_val_split.json', 'w') as f:
    json.dump(val_data, f)

print("Train and validation JSON files created successfully.")
