import argparse
from ultralytics import YOLO
import torch
import os
from torch.optim.lr_scheduler import CyclicLR
from torch.optim import Adam


num_gpus = torch.cuda.device_count()

if num_gpus > 0:
    device = ",".join(str(i) for i in range(num_gpus))  # Use all GPUs
else:
    device = "cpu"

# Set up argument parsing
parser = argparse.ArgumentParser(description="Train and Test YOLO model.")
parser.add_argument("--yaml", required=True, help="Path to the dataset YAML file.")
parser.add_argument("--model", required=True, help="Path to the dataset YAML file.")
parser.add_argument("--output_dir", required=True, help="Directory where YOLO training results will be saved.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Load the M model
model_type = args.model
model = YOLO(model_type)
print(f"Model is {model_type}")
print(f"Yaml file: {args.yaml}")
print(f"Output directory: {args.output_dir}")

# optimizer = model.optimizer

base_lr = 0.0001  # Lower bound for learning rate
max_lr = 0.001    # Upper bound for learning rate
step_size_up = 2000  # Number of iterations to go from base_lr to max_lr
# pytorch_model = model.model
# optimizer = Adam(pytorch_model.parameters(), lr=base_lr)
# print(f"optimizer = {optimizer}")
# scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, mode='triangular', cycle_momentum=False)

# def on_train_batch_end(trainer):
#     scheduler.step()  # Adjust learning rate after each batch
#     current_lr = optimizer.param_groups[0]['lr']
#     print(f"Updated Learning Rate: {current_lr:.6f}")

# model.add_callback("on_train_batch_end", on_train_batch_end)


train_results = model.train(
    data=args.yaml,  
    epochs=500, 
    imgsz=640, 
    optimizer="SGD",
    lr0 = 0.001, # tr lr 0.001, ft lr 0.0007
    device=device, 
    patience=100,
    # freeze=10,
    project=args.output_dir, 
    seed=44  # 0 for it1, 2 for it2, 3 for it3, 4 and 5 and 6 for finetuning, 10 for corrupt training

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
