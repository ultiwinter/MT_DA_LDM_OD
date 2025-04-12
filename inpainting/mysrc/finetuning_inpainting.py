import argparse
import itertools
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder, insecure_hashlib
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

check_min_version("0.13.0.dev0")
logger = get_logger(__name__)


def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB")).transpose(2, 0, 1)[None] / 127.5 - 1.0
    mask = np.array(mask.convert("L")).astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)[None, None]
    return torch.from_numpy(mask), torch.from_numpy(image) * (mask < 0.5)


def random_mask(im_shape, ratio=1, mask_full_image=False):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    center = (random.randint(size[0] // 2, im_shape[0] - size[0] // 2), random.randint(size[1] // 2, im_shape[1] - size[1] // 2))
    draw.rectangle([center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2], fill=255)
    return mask


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--class_data_dir", type=str)
    parser.add_argument("--instance_prompt", type=str)
    parser.add_argument("--class_prompt", type=str)
    parser.add_argument("--with_prior_preservation", action="store_true")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--num_class_images", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="text-inversion-model")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str)
    parser.add_argument("--hub_model_id", type=str)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int)
    parser.add_argument("--resume_from_checkpoint", type=str)
    return parser.parse_args()


class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_root, instance_prompt, tokenizer, class_data_root=None, class_prompt=None, size=512, center_crop=False):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self.class_images_path = list(Path(class_data_root).iterdir()) if class_data_root else None
        self.num_class_images = len(self.class_images_path) if self.class_images_path else 0
        self.class_prompt = class_prompt

        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return max(self.num_instance_images, self.num_class_images)

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images]).convert("RGB")
        instance_image = self.image_transforms(instance_image)
        example["PIL_images"] = instance_image
        example["instance_images"] = instance_image
        example["instance_prompt_ids"] = self.tokenizer(self.instance_prompt, truncation=True, max_length=self.tokenizer.model_max_length).input_ids

        if self.class_images_path:
            class_image = Image.open(self.class_images_path[index % self.num_class_images]).convert("RGB")
            class_image = self.image_transforms(class_image)
            example["class_images"] = class_image
            example["class_prompt_ids"] = self.tokenizer(self.class_prompt, truncation=True, max_length=self.tokenizer.model_max_length).input_ids

        return example


class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {"prompt": self.prompt, "index": index}


def main():
    args = parse_args()
    project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=Path(args.output_dir, args.logging_dir))
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision, log_with="tensorboard", project_config=project_config)

    if args.seed:
        set_seed(args.seed)

    if args.with_prior_preservation and len(list(Path(args.class_data_dir).iterdir())) < args.num_class_images:
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16 if accelerator.device.type == "cuda" else torch.float32, safety_checker=None)
        pipeline.to(accelerator.device)
        sample_dataloader = DataLoader(PromptDataset(args.class_prompt, args.num_class_images), batch_size=args.sample_batch_size, num_workers=1)
        sample_dataloader = accelerator.prepare(sample_dataloader)
        transform_to_pil = transforms.ToPILImage()
        for example in tqdm(sample_dataloader, desc="Generating class images"):
            fake_image = transform_to_pil(torch.rand(3, args.resolution, args.resolution))
            fake_mask = random_mask((args.resolution, args.resolution), mask_full_image=True)
            images = pipeline(prompt=example["prompt"], mask_image=fake_mask, image=fake_image).images
            for i, image in enumerate(images):
                image.save(Path(args.class_data_dir, f"{example['index'][i]}-{insecure_hashlib.sha1(image.tobytes()).hexdigest()}.jpg"))
        del pipeline

    if accelerator.is_main_process and args.push_to_hub:
        create_repo(repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token)

    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name or args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()
    if args.scale_lr:
        args.learning_rate *= args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        masks, masked_images = zip(*[prepare_mask_and_masked_image(example["PIL_images"], random_mask(example["PIL_images"].size)) for example in examples])
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            masks += masks
            masked_images += masked_images
        return {
            "input_ids": tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids,
            "pixel_values": torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float(),
            "masks": torch.stack(masks),
            "masked_images": torch.stack(masked_images),
        }

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if not args.max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes, num_training_steps=args.max_train_steps * accelerator.num_processes)

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, text_encoder, optimizer, train_dataloader, lr_scheduler)
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)
    accelerator.register_for_checkpointing(lr_scheduler)

    weight_dtype = torch.float32 if args.mixed_precision == "no" else torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    accelerator.init_trackers("dreambooth", config=vars(args))

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint if args.resume_from_checkpoint != "latest" else sorted([d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")], key=lambda x: int(x.split("-")[1]))[-1]
        if path:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                masked_latents = vae.encode(batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                mask = F.interpolate(batch["masks"], size=(args.resolution // 8, args.resolution // 8))
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample
                target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
                if args.with_prior_preservation:
                    noise_pred, noise_pred_prior = noise_pred.chunk(2, dim=0)
                    target, target_prior = target.chunk(2, dim=0)
                    loss = F.mse_loss(noise_pred.float(), target.float()).mean() + args.prior_loss_weight * F.mse_loss(noise_pred_prior.float(), target_prior.float()).mean()
                else:
                    loss = F.mse_loss(noise_pred.float(), target.float()).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            accelerator.log({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, unet=accelerator.unwrap_model(unet), text_encoder=accelerator.unwrap_model(text_encoder))
        pipeline.save_pretrained(args.output_dir)
        if args.push_to_hub:
            upload_folder(repo_id=args.hub_model_id or Path(args.output_dir).name, folder_path=args.output_dir, commit_message="End of training", ignore_patterns=["step_*", "epoch_*"])

    accelerator.end_training()


if __name__ == "__main__":
    main()
