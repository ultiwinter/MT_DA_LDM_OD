from ultralytics import YOLOv10

from ultralytics.data.converter import convert_coco

# convert_coco(labels_dir="/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolo/yolo_format/annotations/", save_dir="/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolo/real_yolo_format")

# print(f"Coversion from coco to yolo format done!")
# # /home/woody/iwi5/iwi5215h/masterarbeit/repos/yolov10/coco_converted

model = YOLOv10()
# If you want to finetune the model with pretrained weights, you could load the 
# pretrained weights like below
model = YOLOv10.from_pretrained('jameslahm/yolov10n')


model.train(data='/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/yolo/yolo_format/data.yaml', epochs=500, batch=256)