import argparse, os, sys, glob
sys.path.append(os.getcwd()+"/ldm")
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torch.utils.data import Dataset, DataLoader
from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from ldm.models.autoencoder import AutoencoderKL

# config_ldm = OmegaConf.load("models/ldm/inpainting_big/config.yaml")  # or configs/stable-diffusion/v1-inpainting-inference.yaml  or models/ldm/inpainting_big/config.yaml
# config_auto = OmegaConf.load("models/first_stage_models/vq-f4-noattn/config.yaml")


# ldm_model_path = "models/ldm/inpainting_big/last.ckpt"  # or models/ldm/inpainting_big/model.ckpt  or models/ldm/inpainting_big/last.ckpt
# auto_model_path = "auto.ckpt"  ## noch abgeklaert werden muss


# ##load state dict of autoencoder
# autoencoder_dict=torch.load(auto_model_path)['state_dict']


# ##create ldm model and load weight
# model = instantiate_from_config(config_ldm.model)
# model.load_state_dict(torch.load(ldm_model_path)['state_dict'],strict=False)


# ##get the weight as a dictinoary
# cache_state_dict=model.state_dict()


# ##update the weight dictionary
# for key in cache_state_dict:
#     if 'first_stage_model' in key or 'cond_stage_model' in key :
#         print(key)
#         state_name=key.split('.')[1:]
#         state='.'.join(state_name)
        
#         cache_state_dict[key] = autoencoder_dict[state]


# ldm_model = torch.load(ldm_model_path)
# ldm_model.keys()

# ## load a updated state_dict 
# model.load_state_dict(cache_state_dict)


# ## save the updated model
# torch.save({
#             'epoch': ldm_model['epoch'],
#             'global_step': ldm_model['global_step'],
#             'pytorch-lightning_version': ldm_model['pytorch-lightning_version'],
#             'state_dict': model.state_dict(),
#             'callbacks': ldm_model['callbacks'],
#             'optimizer_states': ldm_model['optimizer_states'],
#             'lr_schedulers': ldm_model['lr_schedulers'],
#             }, "updated_ldm.ckpt")


config_ldm = OmegaConf.load("configs/stable-diffusion/v1-inpainting-inference.yaml")
ldm_model_path = "models/ldm/inpainting_big/model_ft.ckpt"
model1 = instantiate_from_config(config_ldm.model)
model1.load_state_dict(torch.load(ldm_model_path)['state_dict'], strict=True)

# ae = AutoencoderKL()
# clip_embedder = FrozenCLIPEmbedder()

model2 = instantiate_from_config(config_ldm.model)

model1_dict = model1.state_dict()
model2_dict = model2.state_dict()
if model1_dict.keys() != model2_dict.keys():
    print(f"YESSSSS WUPPA WUPPA DUB DUB")
else:
    print(f"Damn it Morty!")