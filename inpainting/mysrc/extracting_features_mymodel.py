import sys
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from main import instantiate_from_config
from torchvision import transforms
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

from data_analysis import CocoAnalyzer
from ldm.models.diffusion.ddim import DDIMSampler
from transformers import CLIPProcessor, CLIPModel


config = "configs/stable-diffusion/v1-inpainting-inference.yaml"
ckpt = "models/ldm/inpainting_big/model.ckpt"

MAX_SIZE = 640

def get_text_embeddings(model, texts, device):
    print(f"Extracting textual embeddings...")
    with torch.no_grad():
        tokenizer = model.cond_stage_model.tokenizer
        text_tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        text_embeddings = model.cond_stage_model.transformer(text_tokens).last_hidden_state
        # text_embeddings = model.cond_stage_model.proj(text_embeddings)
        text_embeddings = text_embeddings.mean(dim=1)
    return text_embeddings

def initialize_model(config, ckpt):
    print(f"Initalizaing the pretrained model...")
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    return model

def grid_image(image, grid_size):
    if grid_size == 0:
        return image
    
    width, height = image.size
    grid_w, grid_h = width // grid_size, height // grid_size
    patches = []

    for i in range(0, width, grid_w):
        for j in range(0, height, grid_h):
            patch = image.crop((i, j, i + grid_w, j + grid_h))
            patches.append((patch, (i, j, grid_w, grid_h)))
    return patches

def extract_features(image, model, device):
    print(f"Extracting features from the image...")
    transform = transforms.Compose([
        transforms.Resize((model.image_size, model.image_size)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_first_stage(image)
    # if isinstance(features, tuple):
    #     features = features[0]
    return features.mean

def classify_regions(image_path, model, processor, clip_model, text_labels, grid_size=0):
    image = Image.open(image_path)
    # patches = grid_image(image, grid_size)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # for patch, (x, y, w, h) in tqdm(patches, desc="Processing regions"):
    #     features = extract_features(patch, model, device)
    #     text_inputs = processor(text=text_labels, return_tensors="pt", padding=True)
    #     text_features = clip_model.get_text_features(**text_inputs)

    #     # flattening features if needed
    #     if features.dim() > 2:
    #         features = features.view(features.size(0), -1)

    #     # adjusting dimensions to match for matrix multiplication
    #     if features.size(1) != text_features.size(1):
    #         features = torch.nn.functional.adaptive_avg_pool1d(features.unsqueeze(0), text_features.size(1)).squeeze(0)


    #     similarity = torch.matmul(features, text_features.T)
    #     probs = similarity.softmax(dim=-1)

    #     top_prob, top_idx = probs.max(dim=-1)
    #     predicted_class = text_labels[top_idx.item()]

    features = extract_features(image, model, device)
    # text_inputs = processor(text=text_labels, return_tensors="pt", padding=True)
    # text_features = clip_model.get_text_features(**text_inputs)
    text_features = get_text_embeddings(model, text_labels, device)

    # flattening features if needed
    if features.dim() > 2:
        features = features.view(features.size(0), -1)

    # adjusting dimensions to match for matrix multiplication
    if features.size(1) != text_features.size(1):
        features = torch.nn.functional.adaptive_avg_pool1d(features.unsqueeze(0), text_features.size(1)).squeeze(0)


    similarity = torch.matmul(features, text_features.T)
    probs = similarity.softmax(dim=-1)

    top_prob, top_idx = probs.max(dim=-1)
    predicted_class = text_labels[top_idx.item()]

    return predicted_class

def run():
    config = "configs/stable-diffusion/v1-inpainting-inference.yaml"
    ckpt = "models/ldm/inpainting_big/model.ckpt"
    model = initialize_model(config, ckpt)
    image_path = "demo-images/demo_dog_165388.png"
    label_analyzer = CocoAnalyzer("/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/instances_all.json")
    text_labels = label_analyzer.get_all_class_names()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    predicted_class = classify_regions(image_path, model, processor, clip_model, text_labels, grid_size=0)

    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":    
    run()
