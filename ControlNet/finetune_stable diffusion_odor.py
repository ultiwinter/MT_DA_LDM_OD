from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from odor_controlnet_dataset import ODORDatasetCN
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from datetime import datetime



timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

resume_path = './models/control_sd15_hed.pth'
batch_size = 10
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
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
    filename=f"finetuned-controlnet-{timestamp}-{{epoch:02d}}",
    save_top_k=1,                                     
)


dataset = ODORDatasetCN()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback], max_epochs=5)


print(f"Now Training ControlNet for ODOR images starts!")
trainer.fit(model, dataloader)
