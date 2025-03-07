import argparse
import logging
import math
import os
import yaml
import shutil
import datetime
import json

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers.utils import ContextManagers
from PIL import Image
from copy import deepcopy
from safetensors import safe_open

import diffusers
from diffusers import DDPMScheduler, LCMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from models import SAR2OptUNetv2, SAR2OptUNetv3
from dataloader_pair import pair_Dataset, pair_Dataset_csv
from utils import update_args_from_yaml, safe_load
from torchvision.utils import make_grid, save_image
from color_loss import Blur

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--config_yaml", 
                        type=str, 
                        default="configs.yaml",
                        help='Path to the YAML configuration file')
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--sar_image_path",
        type=str,
        default=None,
        help=(
            "A folder containing the training data of SAR."
        ),
    )
    parser.add_argument(
        "--opt_image_path",
        type=str,
        default=None,
        help=(
            "A folder containing the training data of OPT."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X epochs.",
    )
    parser.add_argument("--validation_ouputdir", type=str, default='debug')
    parser.add_argument("--loss_type", type=str, default='l2',choices=["l2", "huber"])
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter. Only used if `--loss_type=huber`.",
    )
    parser.add_argument("--num_ddim_timesteps",type=int, default=50)
    parser.add_argument(
    "--timestep_scaling_factor",
    type=float,
    default=10.0,
    help=(
        "The multiplicative timestep scaling factor used when calculating the boundary scalings for LCM. The"
        " higher the scaling is, the lower the approximation error, but the default value of 10.0 should typically"
        " suffice."
    ),)
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.95,
        required=False,
        help="The exponential moving average (EMA) rate or decay factor.",
    )
    parser.add_argument("--unet_checkpoint",
                        type=str,
                        default="diffusion.safetensors",
                        help='Path to the pre-trained diffusion')

    return parser


def inference(model,lcm_scheduler,samples, batch_val):
    device = model.device
    model.eval()
    num_inference_steps = 8
    lcm_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = lcm_scheduler.timesteps
    with torch.no_grad():
        latents = samples
        for timestep in tqdm(timesteps, desc=f"Inference:"):
            latent_to_pred = torch.cat((latents, batch_val), dim=1)
            model_pred = model(latent_to_pred, timestep)
            latents, denoised = lcm_scheduler.step(model_pred, timestep, latents, return_dict=False)

    model.train()
    return denoised

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev

@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out

# Compare LCMScheduler.step, Step 4
def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0

# Based on step 4 in DDIMScheduler.step
def get_predicted_noise(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_epsilon

logger = get_logger(__name__, log_level="INFO")
parser = parse_args()
args = parser.parse_args()
# yaml 更新训练参数等
with open(args.config_yaml, 'r') as f:
    yaml_args = yaml.safe_load(f)
update_args_from_yaml(yaml_args, args, parser)
os.makedirs(args.output_dir, exist_ok=True)

logging_dir = os.path.join(args.output_dir, args.logging_dir)
now = datetime.datetime.now()
trail_name = now.strftime("%m%d-%H%M")
yaml_save_path = os.path.join(args.output_dir, f"{trail_name}.yaml")  # 替换为你想要保存的目录和文件名
with open(yaml_save_path, 'w') as f:
    yaml.dump(yaml_args, f)

accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger.info(accelerator.state, main_process_only=False)

if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

if args.seed is not None:
    set_seed(args.seed)

noise_scheduler = DDPMScheduler.from_pretrained(os.path.join(args.pretrained_model_path,"scheduler"))
alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
solver = DDIMSolver(
    noise_scheduler.alphas_cumprod.numpy(),
    timesteps=noise_scheduler.config.num_train_timesteps,
    ddim_timesteps=args.num_ddim_timesteps,
)

lcm_scheduler = LCMScheduler(num_train_timesteps=1000)

unet = SAR2OptUNetv3(
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
unet.train()
# unet_checkpoint = "/path/to/pretrained/unet"
unet.load_state_dict(safe_load(args.unet_checkpoint), strict=True)

teacher_unet = deepcopy(unet)
teacher_unet.requires_grad_(False)

target_unet = SAR2OptUNetv3(
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
target_unet.load_state_dict(unet.state_dict())
target_unet.train()
target_unet.requires_grad_(False)

if args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

if args.scale_lr:
    args.learning_rate = (
        args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    )

optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

transform_sar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((args.resolution, args.resolution)),
    transforms.Normalize((0.5), (0.5)),
])

transform_opt = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((args.resolution, args.resolution)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = pair_Dataset_csv(args.sar_image_path, args.opt_image_path, transform_sar, transform_opt)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=args.train_batch_size,
    num_workers=args.dataloader_num_workers,
)

# Scheduler and math around the number of training steps.
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    num_training_steps=args.max_train_steps * accelerator.num_processes,
)

unet,optimizer, lr_scheduler = accelerator.prepare(
    unet, optimizer, lr_scheduler
)
alpha_schedule = alpha_schedule.to(accelerator.device)
sigma_schedule = sigma_schedule.to(accelerator.device)
solver = solver.to(accelerator.device)
target_unet.to(accelerator.device)
teacher_unet.to(accelerator.device)

weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
    args.mixed_precision = accelerator.mixed_precision
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16
    args.mixed_precision = accelerator.mixed_precision

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if overrode_max_train_steps:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

if args.validation_steps > 0:
    images = []
    images_opt = []
    image_val = ["/path/to/val/images"]
    for image_path in image_val:
        image = Image.open(image_path)
        image_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])(image)
        images.append(image_tensor)
    batch_val = torch.stack(images, dim=0)
    os.makedirs(args.validation_ouputdir, exist_ok=True)
    image_opt = ["/path/to/val/images"]
    
    for image_path in image_opt:
        image = Image.open(image_path)
        image_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])(image)
        images_opt.append(image_tensor)
    batch_opt = torch.stack(images_opt, dim=0)
    save_image(batch_val, os.path.join(args.validation_ouputdir, f'recon_image_org_sar.png'),
                                   normalize=True, value_range=(-1, 1), nrow=2)
    save_image(batch_opt, os.path.join(args.validation_ouputdir, f'recon_image_org_opt.png'),
                                   normalize=True, value_range=(-1, 1), nrow=2)


# Train!
total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

logger.info("***** Running training *****")
logger.info(f"  Num examples = {len(train_dataset)}")
logger.info(f"  Num Epochs = {args.num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {args.max_train_steps}")
global_step = 0
first_epoch = 0
initial_global_step = 0
blur_rgb = Blur(3)
blur_rgb.to(accelerator.device)

progress_bar = tqdm(
    range(0, args.max_train_steps),
    initial=initial_global_step,
    desc="Steps",
    disable=not accelerator.is_local_main_process,
)

for epoch in range(first_epoch, args.num_train_epochs):
    train_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet):
            sar_image, opt_image = batch
            sar_image, opt_image = sar_image.to(accelerator.device, non_blocking=True), opt_image.to(accelerator.device, non_blocking=True)
            latents = opt_image

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
            index = torch.randint(0, args.num_ddim_timesteps, (bsz,), device=latents.device).long()
            start_timesteps = solver.ddim_timesteps[index]
            timesteps = start_timesteps - topk
            timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

            c_skip_start, c_out_start = scalings_for_boundary_conditions(
                    start_timesteps, timestep_scaling=args.timestep_scaling_factor
                )
            c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
            c_skip, c_out = scalings_for_boundary_conditions(
                timesteps, timestep_scaling=args.timestep_scaling_factor
            )
            c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

            noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)
            noisy_model_input_cat = torch.cat((noisy_model_input, sar_image), dim=1)

            # Get the target for loss depending on the prediction type
            if args.prediction_type is not None:
                # set prediction_type of scheduler if defined
                noise_scheduler.register_to_config(prediction_type=args.prediction_type)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            noise_pred = unet(noisy_model_input_cat, start_timesteps)

            pred_x_0 = get_predicted_original_sample(
                    noise_pred,
                    start_timesteps,
                    noisy_model_input,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )
            model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

            with torch.no_grad():
                with torch.autocast(accelerator.device.type):
                    teacher_output = teacher_unet(
                            noisy_model_input_cat.to(weight_dtype),
                            start_timesteps
                        )
                    pred_x0 = get_predicted_original_sample(
                            teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                    pred_noise = get_predicted_noise(
                            teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            noise_scheduler.config.prediction_type,
                            alpha_schedule,
                            sigma_schedule,
                        )
                    x_prev = solver.ddim_step(pred_x0, pred_noise, index)
            
            with torch.no_grad():
                with torch.autocast(accelerator.device.type):
                    x_prev_cat = torch.cat((x_prev, sar_image), dim=1)
                    target_noise_pred = target_unet(
                            x_prev_cat.float(),
                            timesteps
                        )
                pred_x_0 = get_predicted_original_sample(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        noise_scheduler.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )
                target = c_skip * x_prev + c_out * pred_x_0
            if args.loss_type == "l2":
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            elif args.loss_type == "huber":
                loss = torch.mean(
                    torch.sqrt((model_pred.float() - target.float()) ** 2 + args.huber_c**2) - args.huber_c
                )
            # blur_rgb1 = blur_rgb(pred_x_0.to(torch.float))
            # blur_rgb2 = blur_rgb(latents)
            # color_l = F.mse_loss(blur_rgb1, blur_rgb2)
            # loss = loss * 0.4 + color_l * 0.6
            # Backpropagate
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        accelerator.wait_for_everyone()
        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            update_ema(target_unet.parameters(), unet.parameters(), args.ema_decay)
            progress_bar.update(1)
            global_step += 1

            if (global_step - 1) % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    target_unet.save_pretrained(os.path.join(args.output_dir, f"checkpoint_ema-{global_step}"))    
                    logger.info(f"Saved state to {save_path}")
                accelerator.wait_for_everyone()
        with torch.no_grad():
            if accelerator.sync_gradients and (global_step - 1) % args.validation_steps == 0:
                torch.cuda.empty_cache()
                latent = torch.randn(
                            size=[4, 3, 256, 256], #4 //8
                            device=accelerator.device,
                            generator=torch.Generator(accelerator.device).manual_seed(3407))
                batch_val = batch_val.to(accelerator.device)


                samples_ema = inference(target_unet, lcm_scheduler,latent, batch_val)  
                save_image(samples_ema.to(accelerator.device),
                            os.path.join(args.validation_ouputdir, f'debug_images_ema_{global_step}.jpg'),
                            normalize=True, value_range=(-1, 1), nrow=4)            
                
                samples = inference(unet, lcm_scheduler,latent, batch_val)                    
                save_image(samples.to(accelerator.device),
                                os.path.join(args.validation_ouputdir, f'debug_images_{global_step}.jpg'),
                                normalize=True, value_range=(-1, 1), nrow=4)
                del samples, samples_ema

                accelerator.wait_for_everyone()
        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break

accelerator.end_training()
