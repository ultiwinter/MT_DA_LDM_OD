import json
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from collections import defaultdict
import random

# Load the COCO annotations
with open('/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/instances_train.json') as f:
    coco_data = json.load(f)

# Separate annotations by category
annotations_by_category = defaultdict(list)
for annotation in coco_data['annotations']:
    annotations_by_category[annotation['category_id']].append(annotation)

# Initialize lists to hold the final train and validation data
train_annotations = []
val_annotations = []
train_image_ids = set()
val_image_ids = set()

# Perform the 80-20 split for each category
for category_id, annotations in annotations_by_category.items():
    random.shuffle(annotations)  # Shuffle the annotations to ensure randomness
    split_point = int(len(annotations) * 0.8)  # Determine the split point
    train_annotations.extend(annotations[:split_point])
    val_annotations.extend(annotations[split_point:])

# Collect unique image IDs for train and val splits
train_image_ids.update([ann['image_id'] for ann in train_annotations])
val_image_ids.update([ann['image_id'] for ann in val_annotations])

# Prepare the train and validation image lists
train_images = [img for img in coco_data['images'] if img['id'] in train_image_ids]
val_images = [img for img in coco_data['images'] if img['id'] in val_image_ids]

# Prepare the final COCO json structure for train and val
train_coco_data = {
    'images': train_images,
    'annotations': train_annotations,
    'categories': coco_data['categories']
}

val_coco_data = {
    'images': val_images,
    'annotations': val_annotations,
    'categories': coco_data['categories']
}

# Save the new annotations files
with open('/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_train_split.json', 'w') as f:
    json.dump(train_coco_data, f)

with open('/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_val_split.json', 'w') as f:
    json.dump(val_coco_data, f)
