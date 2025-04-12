import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from myutils import filter_bboxes, convert_bbox_format, filter_degenerate_bboxes

class ODORDataset(Dataset):
    def __init__(self, root, ann_file, transform=None, patch_size=(256, 256)):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]

        path = img_info['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        num_objs = len(anns)

        boxes = []
        labels = []
        iscrowd = []
        area = []
        for i in range(num_objs):
            xmin = anns[i]['bbox'][0]
            ymin = anns[i]['bbox'][1]
            xmax = xmin + anns[i]['bbox'][2]
            ymax = ymin + anns[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(anns[i]['category_id'])
            iscrowd.append(anns[i].get('iscrowd', 0))
            area.append(anns[i]['area'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)

        # Extract patches
        patches, patch_targets = self.extract_patches(img, boxes, labels, iscrowd, area)

        if self.transform:
            patches = [self.transform(patch) for patch in patches]

        return patches, patch_targets

    def extract_patches(self, img, boxes, labels, iscrowd, area):
        patches = []
        patch_targets = []

        img_width, img_height = img.size
        patch_width, patch_height = self.patch_size

        for i in range(0, img_height, patch_height):
            for j in range(0, img_width, patch_width):
                patch = img.crop((j, i, j + patch_width, i + patch_height))
                patch_boxes, patch_labels, patch_iscrowd, patch_area = self.get_patch_targets(
                    boxes, labels, iscrowd, area, j, i, patch_width, patch_height
                )
                if len(patch_boxes) > 0:
                    patches.append(patch)
                    patch_targets.append({
                        "boxes": patch_boxes,
                        "labels": patch_labels,
                        "iscrowd": patch_iscrowd,
                        "area": patch_area
                    })

        return patches, patch_targets

    def get_patch_targets(self, boxes, labels, iscrowd, area, x_offset, y_offset, patch_width, patch_height):
        patch_boxes = []
        patch_labels = []
        patch_iscrowd = []
        patch_area = []

        for i in range(len(boxes)):
            box = boxes[i]
            if (box[0] >= x_offset and box[1] >= y_offset and
                box[2] <= x_offset + patch_width and box[3] <= y_offset + patch_height):
                patch_boxes.append([
                    box[0] - x_offset,
                    box[1] - y_offset,
                    box[2] - x_offset,
                    box[3] - y_offset
                ])
                patch_labels.append(labels[i])
                patch_iscrowd.append(iscrowd[i])
                patch_area.append(area[i])

        patch_boxes = torch.as_tensor(patch_boxes, dtype=torch.float32)
        patch_labels = torch.as_tensor(patch_labels, dtype=torch.int64)
        patch_iscrowd = torch.as_tensor(patch_iscrowd, dtype=torch.int64)
        patch_area = torch.as_tensor(patch_area, dtype=torch.float32)

        return patch_boxes, patch_labels, patch_iscrowd, patch_area


class ODORTrainDataset(Dataset):
    def __init__(self, root, ann_file, patches_per_image=10, transform=None, sample_func=None, patch_size=(256, 256)):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.patches_per_image = patches_per_image
        self.transform = transform
        self.sample_func = sample_func
        self.patch_size = patch_size

        self.printing_boxes = None
        self.printing_labels = None
        self.printing_ann_ids = None

    def __len__(self):
        return len(self.ids) * self.patches_per_image
    
    def get_patch_with_labels(self, img, boxes, labels, iscrowd, area, x=0, y=0):
        patch = img.crop((x, y, x + self.patch_size[0], y + self.patch_size[1]))

        if len(labels) > 0:
            
            filtered_boxes, filtered_labels = filter_bboxes(boxes, labels, x, y, self.patch_size[0], self.patch_size[1])
            if self.transform:
                transformed = self.transform(image=np.array(patch), bboxes=boxes, class_labels=labels)
                patch = transformed['image']
                filtered_boxes = np.array(transformed['bboxes'])
                filtered_labels = transformed['class_labels']
            if len(filtered_boxes) > 0:
                area = (filtered_boxes[:, 3] - filtered_boxes[:, 1]) * (filtered_boxes[:, 2] - filtered_boxes[:, 0])
                if area.any() < 0:
                    raise ValueError("Area is negative!! Check the bounding box format!")
            else:
                area = np.empty([0])
        else:
            if self.transform:
                patch = self.transform(image=np.array(patch))['image']
            filtered_boxes = np.empty((0, 4))
            filtered_labels = np.empty((0,))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        targets = {
            'boxes': torch.as_tensor(filtered_boxes, dtype=torch.float32).to(device),
            'labels': torch.as_tensor(filtered_labels, dtype=torch.int64).to(device),
            'image_id': torch.tensor([img.info['id']]).to(device),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64).to(device),
            'area': torch.as_tensor(area, dtype=torch.float32).to(device)
        }
        # pytorch expects image data to be in the format (channels, height, width) instead of (height, width, channels)
        patch_as_tensor = torch.from_numpy(np.transpose(patch, (2, 0, 1))).float() / 255.0
        return patch_as_tensor, targets

    def __getitem__(self, idx):
        idx_image = idx // self.patches_per_image
        img_id = self.ids[idx_image]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        self.printing_ann_ids = ann_ids
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        

        path = img_info['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        classes = list(set(t['category_id'] for t in anns))
        level_dimensions = img.size  # (width, height)

        img.info = {'id': img_id}

        num_objs = len(anns)
        boxes = [ann['bbox'] for ann in anns]
        boxes = np.array(convert_bbox_format(boxes=boxes))
        labels = np.array([ann['category_id'] for ann in anns])
        boxes, labels = self.filter_degenerate_bboxes(boxes, labels)
        iscrowd = np.array([ann.get('iscrowd', 0) for ann in anns])
        area = np.array([ann['area'] for ann in anns])

        # Sampling a patch
        if self.sample_func:
            x, y = self.sample_func(
            anns, labels, self.patch_size, level_dimensions
        )
        else:
            x = np.random.randint(0, max(1, img.width - self.patch_size[0]))
            y = np.random.randint(0, max(1, img.height - self.patch_size[1]))

        return self.get_patch_with_labels(img, boxes, labels, iscrowd, area, x, y)


    def filter_degenerate_bboxes(self, bboxes, labels):

        if len(bboxes) != len(labels):
            raise ValueError("Lists 'bboxes' and 'labels' should have the same length but have length {} and {}".format(len(bboxes), len(labels)))

        # Identify degenerate bounding boxes
        invalid_indices = np.where((bboxes[:, 2] <= bboxes[:, 0]) | (bboxes[:, 3] <= bboxes[:, 1]))[0]

        if len(invalid_indices) > 0:
            print("Pre-Check: Invalid bboxes found and removed:")
            for idx in invalid_indices:
                print(f"Invalid bbox: [{bboxes[idx,0]}, {bboxes[idx,1]}, {bboxes[idx,2]}, {bboxes[idx,3]}]")
            
            # Remove degenerate bounding boxes
            bboxes = np.delete(bboxes, invalid_indices, axis=0)
            labels = np.delete(labels, invalid_indices, axis=0)

        return bboxes, labels


    
class ODORTestDataset(Dataset):
    def __init__(self, root, ann_file, patch_size=(256, 256), overlap=0.5, transform=None):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.patch_size = patch_size
        self.transform = transform
        self.overlap = overlap
        self.index_dict = {}
        self.my_index_dict = self._create_index_dict()

    def _create_index_dict(self):
        index_dict = {}
        idx = 0
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            num_patches_x = int((img_info['width'] - self.patch_size[0]) / (self.patch_size[0] * self.overlap)) + 1
            num_patches_y = int((img_info['height'] - self.patch_size[1]) / (self.patch_size[1] * self.overlap)) + 1
            for px in range(num_patches_x):
                for py in range(num_patches_y):
                    index_dict[idx] = (img_id, px, py)
                    idx += 1
        return index_dict

    def __len__(self):
        return len(self.index_dict)

    def __getitem__(self, idx):
        img_id, patch_idx_x, patch_idx_y = self.my_index_dict[idx]
        # img_id = self.ids[idx % len(self.ids)]
        img_info = self.coco.loadImgs(img_id)[0]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        path = img_info['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        x = int(patch_idx_x * self.patch_size[0] * self.overlap)
        y = int(patch_idx_y * self.patch_size[1] * self.overlap)

        patch = img.crop((x, y, x + self.patch_size[0], y + self.patch_size[1]))
        self.index_dict[idx] = [x,y] # saving for local_to_global later


        boxes = np.array([ann['bbox'] for ann in anns])
        boxes = np.array(convert_bbox_format(boxes=boxes))
        labels = np.array([ann['category_id'] for ann in anns])
        iscrowd = np.array([ann.get('iscrowd', 0) for ann in anns])
        area = np.array([ann['area'] for ann in anns])

        if len(labels) > 0:
            filtered_boxes, filtered_labels = filter_bboxes(boxes, labels, x, y, self.patch_size[0], self.patch_size[1])
            if self.transform:
                transformed = self.transform(image=np.array(patch), bboxes=filtered_boxes, class_labels=filtered_labels)
                patch = transformed['image']
                filtered_boxes = np.array(transformed['bboxes'])
                filtered_labels = transformed['class_labels']

            if len(filtered_boxes) > 0:
                area = (filtered_boxes[:, 3] - filtered_boxes[:, 1]) * (filtered_boxes[:, 2] - filtered_boxes[:, 0])
            else:
                area = np.empty([0])
        else:
            if self.transform:
                patch = self.transform(image=np.array(patch))['image']
            filtered_boxes = np.empty((0, 4))
            filtered_labels = np.empty((0,))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        targets = {
            'boxes': torch.as_tensor(filtered_boxes, dtype=torch.float32).to(device),
            'labels': torch.as_tensor(filtered_labels, dtype=torch.int64).to(device),
            'image_id': torch.tensor([img_id]).to(device),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64).to(device),
            'area': torch.as_tensor(area, dtype=torch.float32).to(device)
        }

        patch_as_tensor = torch.from_numpy(np.transpose(patch, (2, 0, 1))).float() / 255.0
        return patch_as_tensor, targets
    
    def get_whole_img_labels_as_dict(self, img_id) -> dict:
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        bboxes = np.array([ann['bbox'] for ann in anns])
        bboxes = bboxes.reshape((-1, 4))
        labels = np.array([ann['category_id'] for ann in anns])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        targets = {
            'boxes': torch.as_tensor(bboxes, dtype=torch.float32).to(device),
            'labels': torch.as_tensor(labels, dtype=torch.int64).to(device),
            'image_id': torch.tensor([img_id]).to(device),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64).to(device),
            'area': torch.as_tensor(area, dtype=torch.float32).to(device)
        }

        return targets

    def local_to_global(self, idx, bboxes:torch.Tensor) -> torch.Tensor:
        # TODO: replace the following lines: Transform local patch / bbox coordinates to global (RoI-wise ones)
        i_patch_x, i_patch_y = self.index_dict[idx]
        bboxes_global = bboxes.clone()
        bboxes_global[:,0]+=i_patch_x
        bboxes_global[:,2]+=i_patch_x
        bboxes_global[:,1]+=i_patch_y
        bboxes_global[:,3]+=i_patch_y

        return bboxes_global


# TODO: Define transformations
transform = T.Compose([
    T.ToTensor(),
])

if __name__ == "__main__":
        
    image_folder = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images'

    train_ann_file = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/instances_train.json'
    test_ann_file = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/instances_test.json'

    train_dataset = ODORTrainDataset(root=image_folder, ann_file=train_ann_file, transform=transform)
    test_dataset = ODORTestDataset(root=image_folder, ann_file=test_ann_file, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

    print(f"Dataset works flawlessly without any errors. Nevertheless, that doesn't speak of the functionality!")
