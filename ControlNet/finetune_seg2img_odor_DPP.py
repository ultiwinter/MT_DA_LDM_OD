from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from odor_controlnet_dataset import ODORDatasetCN
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from datetime import datetime
import torch

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


resume_path = './models/control_sd15_seg.pth' # models/control_sd15_hed.pth
batch_size = 16
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

state_dict = model.state_dict()
print(f"Total number of parameters in state_dict: {len(state_dict)}")
for name, param in state_dict.items():
    print(f"{name}: {param.shape}")

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",   
    filename=f"seg2img-finetuned-instr-oilpaint-seg2img-controlnet-b{batch_size}-s256-{timestamp}-{{epoch:02d}}",
    save_top_k=1,                                     
)

dataset = ODORDatasetCN()
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

trainer = pl.Trainer(
    accelerator="gpu",
    devices=num_gpus,                     
    precision=32,
    strategy= "ddp" if num_gpus > 1 else None,
    callbacks=[logger, checkpoint_callback],
    log_every_n_steps=200,
    progress_bar_refresh_rate=200,    
    max_epochs=20
)

print(f"Now Training ControlNet for ODOR images starts!")
trainer.fit(model, dataloader)
