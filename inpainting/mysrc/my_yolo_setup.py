import argparse
from tqdm import tqdm
import yaml
import json
import os
import shutil
import glob
import sys


def has_valid_imagedir(input_dir):
    image_path = f'{input_dir}'
    if not os.path.isdir(image_path):
        return False
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.bmp', '*.png']: 
        image_files.extend(glob.glob(f'{image_path}/{ext}'))
    return len(image_files) > 0

def validate_input(input_dir, json_files):
    if not has_valid_imagedir(input_dir):
        print("Please provide a valid image directory with at least one input image (jpg, jpeg, bmp, png).")
        sys.exit(1)
    for json_file in json_files:
        if not os.path.exists(json_file):
            print(f"COCO annotations file not found: {json_file}")
            sys.exit(1)

def create_yolo_structure(output_dir, name):
    os.makedirs(f'{output_dir}/{name}/images/train', exist_ok=True)
    os.makedirs(f'{output_dir}/{name}/labels/train', exist_ok=True)
    os.makedirs(f'{output_dir}/{name}/images/valid', exist_ok=True)
    os.makedirs(f'{output_dir}/{name}/labels/valid', exist_ok=True)
    os.makedirs(f'{output_dir}/{name}/images/test', exist_ok=True)
    os.makedirs(f'{output_dir}/{name}/labels/test', exist_ok=True)

def create_yaml(output_dir, name, classes):
    dataset_dict = {
        'path': '.',
        'train': f'{name}/images/train',
        'val': f'{name}/images/valid',
        'test': f'{name}/images/test',
        'names': {i: cls for i, cls in enumerate(classes)}
    }
    with open(f'{output_dir}/{name}.yaml', 'w') as f:
        yaml.dump(dataset_dict, f)

def copy_images(input_dir, output_dir, name, img_ids, split, img_id_map):
    print(f'Copying {split} images...')
    for img_id in tqdm(img_ids):
        img_file = img_id_map[img_id]
        src = f'{input_dir}/{img_file}'
        dst = f'{output_dir}/{name}/images/{split}/{img_file}'
        shutil.copyfile(src, dst)

def to_yolo_bbox(bbox, im_w, im_h):
    x, y, w, h = bbox
    return [
        (x + w / 2) / im_w,  # center X
        (y + h / 2) / im_h,  # center Y
        w / im_w,            # width
        h / im_h             # height
    ]

def create_annotations(output_dir, name, split, coco, img_ids):
    img_id_to_annotations = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_annotations:
            img_id_to_annotations[img_id] = []
        img_id_to_annotations[img_id].append(ann)

    img_id_to_images = {img['id']: img for img in coco['images']}
    print(f'Creating {split} labels...')
    for img_id in tqdm(img_ids):
        img = img_id_to_images[img_id]
        annotations = img_id_to_annotations.get(img_id, [])
        label_lines = []
        for ann in annotations:
            category_id = ann['category_id']
            bbox = to_yolo_bbox(ann['bbox'], img['width'], img['height'])
            label_lines.append(f"{category_id} {' '.join(map(str, bbox))}")
        
        label_path = f"{output_dir}/{name}/labels/{split}/{img['file_name'].rsplit('.', 1)[0]}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))

def process_coco_file(coco_file, input_dir, output_dir, name, split):
    with open(coco_file) as f:
        coco = json.load(f)

    classes = [cat['name'] for cat in coco['categories']]
    img_ids = [img['id'] for img in coco['images']]
    img_id_to_file = {img['id']: img['file_name'] for img in coco['images']}

    copy_images(input_dir, output_dir, name, img_ids, split, img_id_to_file)
    create_annotations(output_dir, name, split, coco, img_ids)

    return classes

def convert(args):
    args.output_dir = os.path.join(args.output_dir, args.dataset_name)
    validate_input(args.input_dir, [args.train_json, args.val_json, args.test_json])
    create_yolo_structure(args.output_dir, args.dataset_name)

    train_classes = process_coco_file(args.train_json, args.input_dir, args.output_dir, args.dataset_name, 'train')
    val_classes = process_coco_file(args.val_json, args.input_dir, args.output_dir, args.dataset_name, 'valid')
    test_classes = process_coco_file(args.test_json, args.input_dir, args.output_dir, args.dataset_name, 'test')


    create_yaml(args.output_dir, args.dataset_name, train_classes)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert COCO datasets to YOLO format.')
    parser.add_argument('--input_dir', default='/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images/',  help='Path to source images directory.')
    parser.add_argument('--output_dir', default='/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/', help='Output directory for YOLO dataset.')
    parser.add_argument('--train_json', default='/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json', help='Path to COCO training annotations JSON file.')
    parser.add_argument('--val_json', default='/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_val_split.json', help='Path to COCO validation annotations JSON file.')
    parser.add_argument('--test_json', default='/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/instances_test.json',  help='Path to COCO testing annotations JSON file.')
    parser.add_argument('--dataset_name', default='baseline_orig', help='Dataset name for YOLO output.')
    return parser.parse_args()

def main():
    args = parse_args()
    convert(args)

if __name__ == '__main__':
    main()
