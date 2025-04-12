import argparse
from ultralytics import YOLO
import torch
from torch.optim.lr_scheduler import CyclicLR
from torch.optim import Adam



import torch
import os
import random

class IncrementalAugmentationSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_epochs, total_epochs):
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.total_epochs = total_epochs
        
        self.original_indices = [i for i, img_path in enumerate(dataset.img_files) if not img_path.endswith("_aug.png")]
        self.augmented_indices = [i for i, img_path in enumerate(dataset.img_files) if img_path.endswith("_aug.png")]
    
    def get_augmentation_ratio(self):
        """Gradually increase augmentation probability as training progresses"""
        return min(1.0, self.num_epochs / self.total_epochs)  # linearly scale from 0 → 100%

    def __iter__(self):
        aug_ratio = self.get_augmentation_ratio()
        num_aug_samples = int(len(self.augmented_indices) * aug_ratio) 


        sampled_aug_indices = random.sample(self.augmented_indices, num_aug_samples) if self.augmented_indices else []

        final_indices = self.original_indices + sampled_aug_indices
        random.shuffle(final_indices)
        
        return iter(final_indices)

    def __len__(self):
        return len(self.original_indices) + int(len(self.original_indices) * self.get_augmentation_ratio())





num_gpus = torch.cuda.device_count()

if num_gpus > 0:
    device = ",".join(str(i) for i in range(num_gpus)) 
else:
    device = "cpu"

parser = argparse.ArgumentParser(description="Train and Test YOLO model.")
parser.add_argument("--yaml", required=True, help="Path to the dataset YAML file.")
parser.add_argument("--model", required=True, help="Path to the dataset YAML file.")
args = parser.parse_args()

model_type = args.model
model = YOLO(model_type)
print(f"Model is {model_type}")

print(f"Yaml file: {args.yaml}")


train_results = model.train(
    data=args.yaml,  
    epochs=500, 
    imgsz=640, 
    device=device, 
    patience=100,
    optimizer="Adam",
    seed=4  # 0 for it1, 2 for it2, 3 for it3, 4 and 5 and 6 for finetuning
)


metrics = model.val()

print("metrics.box.map")
print(metrics.box.map) 
print("metrics.box.map50")
print(metrics.box.map50)
print("metrics.box.map75")
print(metrics.box.map75)
print("metrics.box.maps")
print(metrics.box.maps) 



path = model.export(format="onnx") 

print(f"Path of the trained model: {path}")


# results = model("/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_orig/baseline_orig/images/train/STAEDEL_speisekammer-mit-wildbret.jpg")
# results[0].show()


print(f"TESTING THE MODEL")
metrics = model.val(data="/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolov11/baseline_orig/baseline_orig.yaml", imgsz=640, split="test", save=True)


print("metrics.box.map")
print(metrics.box.map)
print("metrics.box.map50")
print(metrics.box.map50)
print("metrics.box.map75")
print(metrics.box.map75)
print("metrics.box.maps")
print(metrics.box.maps)

print(metrics)  # mAP, precision, recall
