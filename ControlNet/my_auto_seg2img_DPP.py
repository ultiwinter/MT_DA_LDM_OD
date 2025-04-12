import cv2
import einops
import numpy as np
import torch
import random
import os
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3, my_resize_image, modified_resize_image
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config
import json


apply_uniformer = UniformerDetector()

class ImageDataset(Dataset):
    def __init__(self, image_files, input_dir):
        self.image_files = image_files
        self.input_dir = input_dir

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.input_dir, image_file)
        return image_path, image_file

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Set device for each process

def cleanup():
    dist.destroy_process_group()

def load_ddp_model(rank):
    model = create_model('./models/cldm_v15.yaml').to(rank)
    model.load_state_dict(load_state_dict('./models/control_sd15_seg.pth', location=f'cuda:{rank}'))
    model = model.to(rank)
    model.cond_stage_model = model.cond_stage_model.to(rank)

    model = DDP(model, device_ids=[rank])
    ddim_sampler = DDIMSampler(model.module)
    return model, ddim_sampler

def process_image(rank, model, ddim_sampler, input_image_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        # print(f"Currently augmenting {input_image_path}")
        input_image = cv2.imread(input_image_path)
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        apply_uniformer.model = apply_uniformer.model.to(rank)

        # use the default HEDdetector
        detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map).float().to(rank) / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        prompt_text = [prompt + ', ' + a_prompt] * num_samples
        n_prompt_text = [n_prompt] * num_samples

        cond = {
            "c_concat": [control],
            "c_crossattn": [model.module.get_learned_conditioning(prompt_text).to(rank)]
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.module.get_learned_conditioning(n_prompt_text).to(rank)]
        }

        shape = (4, H // 8, W // 8)
        model.module.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

        samples, intermediates = ddim_sampler.sample(
            ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
            unconditional_guidance_scale=scale, unconditional_conditioning=un_cond
        )

        x_samples = model.module.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        return [255 - detected_map] + [x_samples[i] for i in range(num_samples)]

def process_directory(rank, world_size, input_dir, output_dir, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    setup(rank, world_size)
    model, ddim_sampler = load_ddp_model(rank)

    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])  # Sort files to ensure order
    start_index = 2600*9 # CHECK
    print(f"start_index = {start_index}")
    end_index =  2600*10 # CHECK
    print(f"end_index = {end_index}")
    image_files = image_files[start_index:end_index]  # start from the desired index, dividing by 2600 intervals

    dataset = ImageDataset(image_files, input_dir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    data_loader = DataLoader(dataset, batch_size=1, sampler=sampler)

    from class_imbalance_ann import ClassAugmentationPrioritizer
    augmentations_priortizer = ClassAugmentationPrioritizer(annotation_file='/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json')
    augs_dict = augmentations_priortizer.get_category_augmentation_dict()

    for image_path, image_file in tqdm(data_loader, desc=f"Processing images on GPU {rank}"):
        image_path = image_path[0]  # Unpack batch dimension
        class_name = os.path.splitext(image_path)[0].split('_')[-3]
        if class_name == "amphibia":
            class_name = "reptile/amphibia"
        num_augs = augs_dict[class_name]
        if class_name=="reptile/amphibia":
            class_name="reptile-amphibia"
        # print(f"class_name = {class_name}")
        # print(f"num_augs = {num_augs}")
        num_samples = 1 ####################################################### 1 is expermental
        prompt = f"oil painting of {class_name} on canvas"
        a_prompt = ""
        n_prompt = "bad anatomy, bad structure"

        ann_id = int(os.path.splitext(image_path)[0].split('_')[-2])
        with open('/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/my_modified_train_split.json', 'r') as file:
            coco_data = json.load(file)
        width = None
        height = None
        for annotation in coco_data.get("annotations", []):
            if annotation["id"] == ann_id:
                bbox = annotation["bbox"]
                width = bbox[2]
                height = bbox[3]
                break
        if width==None:
            width=128
        if height==None:
            height=128
        print(f"h {height}, w {width}")

        if min(width, height) < 256:
            gen_dim = 256
        elif max(width, height) > 512:
            gen_dim = 512
        else:
            gen_dim = round(min(width, height) / 8) * 8
        
        image_resolution = gen_dim
        detect_resolution = gen_dim

        for aug_idx in range(num_augs):
            current_seed = seed + aug_idx

            results = process_image(
                rank, model, ddim_sampler, image_path, prompt, a_prompt, n_prompt, 1,  #  one augmentation at a time
                image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, 
                current_seed, eta, low_threshold, high_threshold
            )
            seg_maps_dir = os.path.join(os.path.dirname(output_dir), "seg_maps")
            os.makedirs(seg_maps_dir, exist_ok=True)
            for idx, result in enumerate(results):
                if idx==0:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) 
                    output_file = os.path.join(seg_maps_dir, f"{image_file[0].rsplit('.', 1)[0]}_map_{aug_idx + 1}.png")
                    cv2.imwrite(output_file, result)
                else:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # flipped channels back
                    output_file = os.path.join(output_dir, f"{image_file[0].rsplit('.', 1)[0]}_synthesized_{aug_idx + 1}.png")
                    cv2.imwrite(output_file, result)

        # results = process_image(rank, model, ddim_sampler, image_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)

        # for idx, result in enumerate(results[1:]):  # Skip the detected map
        #     result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) # flipped channels
        #     output_file = os.path.join(output_dir, f"{image_file[0].rsplit('.', 1)[0]}_synthesized_{idx + 1}.png")
        #     cv2.imwrite(output_file, result)

    cleanup()


def main():
    world_size = torch.cuda.device_count()
    input_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/bbox_images_train_train_modified'
    output_dir = '/home/woody/iwi5/iwi5215h/masterarbeit/repos/odor-images/FINAL_flip_seg2img_finetuned20_bbox_images_sdm256_181224'
    print(f"Output dir: {output_dir}")
    num_samples = 1
    image_resolution = 64 # 512
    detect_resolution = 64
    ddim_steps = 50
    guess_mode = False
    strength = 1.0
    scale = 7.5
    seed = -1
    eta = 0.0
    low_threshold = 100
    high_threshold = 200

    mp.spawn(
        process_directory,
        args=(world_size, input_dir, output_dir, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
