import sys
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from main import instantiate_from_config
import torch
import os
import argparse
import json
from tqdm import tqdm
import random

from ldm.models.diffusion.ddim import DDIMSampler


MAX_SIZE = 640

# load safety model
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from imwatermark import WatermarkEncoder
import cv2

wm = "StableDiffusionV1-Inpainting"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def put_watermark(img):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
            }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h//8, w//8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        with torch.autocast("cuda"):
            batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

            c = model.cond_stage_model.encode(batch["txt"])

            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    # processing the mask itself
                    bchw = [num_samples, 4, h//8, w//8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    # encoding the masked_image
                    # the following like works for v1-inf and model.ckpt
                    cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                    # option from main inpaint file: cc = model.first_stage_model.encode(cc)
                    # Check if size matches the desired bchw[-2:]
                    if cc.shape[-2:] != tuple(bchw[-2:]):
                        print(f"Interpolation activated!")
                        cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                # print(f"{ck} tensor shape after processing: {cc.shape}")
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond={"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            # print(f"{type(model)=}")
            # print(f"{dir(model)=}")
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            shape = [model.channels, h//8, w//8]
            samples_cfg, intermediates = sampler.sample(
                    ddim_steps,
                    num_samples,
                    shape,
                    cond,
                    verbose=False,
                    eta=1.0,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_full,
                    x_T=start_code,
            )
            x_samples_ddim = model.decode_first_stage(samples_cfg)

            result = torch.clamp((x_samples_ddim+1.0)/2.0,
                                 min=0.0, max=1.0)

            result = result.cpu().numpy().transpose(0,2,3,1)
            result = result*255

    result = [Image.fromarray(img.astype(np.uint8)) for img in result]
    return result


def run():
    parser = argparse.ArgumentParser(description="Stable Diffusion Inpainting")
    parser.add_argument("--config_path", default="configs/stable-diffusion/v1-inpainting-inference.yaml", type=str, help="Path to the config file")  # "models/ldm/inpainting_big/config.yaml"  # configs/stable-diffusion/v1-inpainting-inference.yaml
    parser.add_argument("--checkpoint_path", default="models/ldm/inpainting_big/model.ckpt", type=str, help="Path to the checkpoint file")  # models/ldm/inpainting_big/last_orig.ckpt  # models/ldm/inpainting_big/model.ckpt
    parser.add_argument("--image_dir", type=str, default="/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/images_train_train_modified/", help="Directory containing images")
    parser.add_argument("--mask_dir", type=str, default="/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/object_border_nonoverlap_masks/", help="Directory containing masks")
    parser.add_argument("--annotations_path", type=str, default='/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json', help="Path to the annotations file (COCO format)")
    parser.add_argument("--seed", type=int,default=0, help="Seed for reproducibility")
    parser.add_argument("--scale", type=float, default=7.5, help="Scale for guidance")
    parser.add_argument("--ddim_steps", type=int, default=50, choices=range(1, 50), help="Number of DDIM steps")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/synthesized_borderobj_DA_images/", help="Directory in which the synthetic images are saved")
    args = parser.parse_args()
    
    sampler = initialize_model(args.config_path, args.checkpoint_path)

    with open(args.annotations_path, 'r') as f:
        annotations = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))]
    mask_files = [f for f in os.listdir(args.mask_dir) if os.path.isfile(os.path.join(args.mask_dir, f))]


    # start_idx = 2688
    for image_file in tqdm(image_files,desc="Performing inpainting on ODOR images"):
        image_path = os.path.join(args.image_dir, image_file)
        image = Image.open(image_path)

        width, height = image.size
        if max(width, height) > MAX_SIZE:
            factor = MAX_SIZE / max(width, height)
            width = int(factor * width)
            height = int(factor * height)
        width, height = map(lambda x: x - x % 64, (width, height))
        image = image.resize((width, height))

        corresponding_masks = [m for m in mask_files if m.startswith(os.path.splitext(image_file)[0])]

        nonobj_classes = ["Knights", "Children", "Bards" ,"Coat of arms", "Crown", "Tiara", "Armor", "Harp", "Violin", "Lute", "Throne", "Fountain", "Quill", "Tapestry", "Shield", "Helmet", "Banner", "Cloak", "Lyre", "Hourglass", "Column", "Pedestal", "Arch", "Spyglass", "Carriage", "Gauntlet", "Cameo", "Relief", "Statue", "Tunic", "Fresco", "Bas-relief"]

        for mask_file in corresponding_masks:
            mask_path = os.path.join(args.mask_dir, mask_file)
            mask = Image.open(mask_path)
            mask = mask.resize((width, height))

            class1 = random.choice(nonobj_classes)
            class2 = random.choice(nonobj_classes)
            class3 = random.choice(nonobj_classes)

            prompt = f"oil painting of {class1}, {class2}, {class3} on canvas"


            mask = np.array(mask)
            if len(mask.shape) == 3 and mask.shape[2] == 4:
                mask = mask[:, :, -1]
            mask = mask > 0
            mask = Image.fromarray(mask)

            result = inpaint(
                sampler=sampler,
                image=image,
                mask=mask,
                prompt=prompt,
                seed=args.seed,
                scale=args.scale,
                ddim_steps=args.ddim_steps,
                num_samples=args.num_samples,
                h=height, w=width
            )

            base_name = os.path.splitext(image_file)[0]
            if args.num_samples==1:
                for img in result:
                    new_image_name = f"{base_name}_synthesized.png"
                    img.save(os.path.join(args.output_dir, new_image_name))
            else:
                for index, img in enumerate(result):
                    new_image_name = f"{base_name}_synthesized-{index}.png"
                    img.save(os.path.join(args.output_dir, new_image_name))
    
    print(f"args.output_dir = {args.output_dir}")

if __name__ == "__main__":    
    run()

