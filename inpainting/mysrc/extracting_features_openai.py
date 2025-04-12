import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import cv2
import numpy as np
from skimage import io, img_as_ubyte
from skimage.segmentation import felzenszwalb
from skimage.measure import regionprops
from skimage.color import rgb2gray
from tqdm import tqdm

from data_analysis import CocoAnalyzer

# Load the pre-trained CLIP model
model_name = "openai/clip-vit-large-patch14"  # openai/clip-vit-large-patch14 models/ldm/inpainting_big/model.ckpt
# /home/hpc/iwi5/iwi5215h/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name) 
print(f"Pretrained model and processor loaded successfully!")

def classify_regions(image_path, model, processor, text_labels, num_regions=30):
    image = io.imread(image_path)
    image_pil = Image.fromarray(image)

    inputs = processor(images=image_pil, return_tensors="pt")

    # get the image embedding
    with torch.no_grad():
        # print(f"Extracting image features...")
        image_features = model.get_image_features(**inputs)

    # get the text embeddings (class labels)
    text_inputs = processor(text=text_labels, return_tensors="pt", padding=True)
    with torch.no_grad():
        # print(f"Extracting textual features...")
        text_features = model.get_text_features(**text_inputs)

    # calculate the similarity between image and text embeddings
    similarity = torch.matmul(image_features, text_features.T)
    probs = similarity.softmax(dim=-1)

    top_prob, top_idx = probs.max(dim=-1)
    predicted_class = text_labels[top_idx.item()]

    return predicted_class

# demo usage
image_path = "demo-images/demo_dog_165388.png"
label_analyzer = CocoAnalyzer("/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/instances_all.json")
text_labels = label_analyzer.get_all_class_names()
predicted_class = classify_regions(image_path, model, processor, text_labels)
print(f"Predicted class: {predicted_class}")