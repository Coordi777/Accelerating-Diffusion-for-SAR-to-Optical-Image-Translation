import argparse
import logging
import math
import os
import yaml
import shutil
import datetime
import random
from pathlib import Path
import json
from safetensors import safe_open
import PIL.Image as Image

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, LCMScheduler
import gc

from models import SAR2OptUNetv3
from utils import update_args_from_yaml, safe_load
from torchvision.utils import make_grid, save_image
import pandas as pd

transform_sar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5), (0.5)),
])

# 设置随机种子
seed = 3407
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
device = "cuda:2"

unet_config_path = "/path/to/unet/config.json"
with open(unet_config_path) as unet_config_file:
    unet_config = json.load(unet_config_file)
    unet_type = unet_config.pop("type", None)

def safe_load(model_path):
    assert "safetensors" in model_path
    state_dict = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k) 
    return state_dict

unet_checkpoint = "/path/to/unet/checkpoints"
unet_checkpoint_org = "/path/to/org/unet/checkpoints"
# unet_model = SAR2OptUNetv3(**unet_config)
unet_model = SAR2OptUNetv3(
            sample_size=256,
            in_channels=4,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
)

unet_model.load_state_dict(safe_load(unet_checkpoint), strict=True)
print('load unet safetensos done!')
unet_model.eval().to(device)

lcm_scheduler = LCMScheduler(num_train_timesteps=1000)

sample_list = []
file_path = "selected_opt.txt"
with open(file_path, "r") as file:
    lines = file.readlines()

sample_name_list = [line.strip() for line in lines]

nums_step = 16

save_path = f"/path/to/save/step_{nums_step}_v2"
os.makedirs(save_path, exist_ok=True)

with torch.no_grad():
    for i,opt_image_path in enumerate(sample_list):
        filename = os.path.basename(opt_image_path)
        lcm_scheduler.set_timesteps(nums_step, device=device)
        timesteps = lcm_scheduler.timesteps
        pred_latent = torch.randn(size=[1, 3, 256, 256], device=device)
        image = Image.open(opt_image_path)
        image_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        image_tensor = image_tensor.to(device)

        for timestep in tqdm(timesteps, desc=f"Inference:"):
            latent_to_pred = torch.cat((pred_latent, image_tensor), dim=1)
            model_pred = unet_model(latent_to_pred, timestep)
            pred_latent, denoised = lcm_scheduler.step(
                                                    model_output=model_pred,
                                                    timestep=timestep,
                                                    sample=pred_latent,
                                                    return_dict=False)
        samples = denoised.cpu()
        combined_tensor = samples
        save_image(combined_tensor, os.path.join(save_path, f'idx{i}_{filename}_ema.png'),
            normalize=True, value_range=(-1, 1), nrow=3)