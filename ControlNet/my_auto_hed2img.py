import cv2 
import einops
import numpy as np
import torch
import random
import os
from tqdm import tqdm

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config

# Initialize the model and other components
apply_hed = HEDdetector()

# Load your model
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_hed.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process_image(input_image_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        # Read image from the given path
        input_image = cv2.imread(input_image_path)
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        # Apply HED edge detection
        detected_map = apply_hed(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        # Convert and prepare the control tensor
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # Prepare conditional and unconditional inputs
        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        # Sample from the model
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # Decode and process the output samples
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        # Return detected map and the results as a list of images
        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


def process_directory(input_dir, output_dir, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    start_idx = 2749
    for image_file in tqdm(image_files[start_idx:], desc="Processing data augmentation on bbox images"):
        image_path = os.path.join(input_dir, image_file)
        class_name = os.path.splitext(image_path)[0].split('_')[-3]
        prompt = f"photo of {class_name} on canvas"
        a_prompt = ""
        n_prompt = "bad anatomy, bad structure"
        results = process_image(image_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)
        
        output_file = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_synthesized.png")
        cv2.imwrite(output_file, results[-1])


# Parameters
input_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/bbox_images_train'
output_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/controlnet_hed_output_bbox_images'
num_samples = 1
image_resolution = 512
detect_resolution = 512
ddim_steps = 50
guess_mode = False
strength = 1.0
scale = 7.5
seed = -1
eta = 0.0
low_threshold = 100
high_threshold = 200

process_directory(input_dir, output_dir, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)
