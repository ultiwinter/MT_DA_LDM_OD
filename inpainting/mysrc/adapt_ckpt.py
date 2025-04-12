import torch

key_mapping = {
    "model.diffusion_model.input_blocks.3.0.op.weight": "model.diffusion_model.input_blocks.3.0.in_layers.0.weight",
    "model.diffusion_model.input_blocks.3.0.op.bias": "model.diffusion_model.input_blocks.3.0.in_layers.0.bias",
    "model.diffusion_model.input_blocks.6.0.op.weight": "model.diffusion_model.input_blocks.6.0.in_layers.0.weight",
    "model.diffusion_model.input_blocks.6.0.op.bias": "model.diffusion_model.input_blocks.6.0.in_layers.0.bias",
    "model.diffusion_model.input_blocks.9.0.op.weight": "model.diffusion_model.input_blocks.9.0.in_layers.0.weight",
    "model.diffusion_model.input_blocks.9.0.op.bias": "model.diffusion_model.input_blocks.9.0.in_layers.0.bias",
    "model.diffusion_model.output_blocks.2.1.conv.weight": "model.diffusion_model.output_blocks.2.1.in_layers.0.weight",
    "model.diffusion_model.output_blocks.2.1.conv.bias": "model.diffusion_model.output_blocks.2.1.in_layers.0.bias",
    "model.diffusion_model.output_blocks.5.2.conv.weight": "model.diffusion_model.output_blocks.5.2.in_layers.0.weight",
    "model.diffusion_model.output_blocks.5.2.conv.bias": "model.diffusion_model.output_blocks.5.2.in_layers.0.bias",
    "model.diffusion_model.output_blocks.8.2.conv.weight": "model.diffusion_model.output_blocks.8.2.in_layers.0.weight",
    "model.diffusion_model.output_blocks.8.2.conv.bias": "model.diffusion_model.output_blocks.8.2.in_layers.0.bias",

}

adapted_ckpt_state_dict = {}
for ckpt_key, ckpt_value in ckpt_state_dict.items():
    new_key = key_mapping.get(ckpt_key, ckpt_key)
    adapted_ckpt_state_dict[new_key] = ckpt_value

model.load_state_dict(adapted_ckpt_state_dict, strict=False)


checkpoint_path = "models/ldm/stable-diffusion-v1/sd-v1-4.ckpt"  # Replace with your checkpoint path

checkpoint = torch.load(checkpoint_path, map_location='cpu')

state_dict = checkpoint["state_dict"]

print("Number of keys before removal:", len(state_dict))

keys_to_remove = ["model_ema.decay", "model_ema.num_updates"]
for key in keys_to_remove:
    if key in state_dict:
        del state_dict[key]

print("Number of keys after removal:", len(state_dict))

torch.save(checkpoint, "modified_checkpoint.ckpt")

state_dict = checkpoint["state_dict"]
print("Number of keys after removal in the end:", len(state_dict))

