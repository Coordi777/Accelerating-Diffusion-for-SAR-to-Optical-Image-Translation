from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, UNet2DModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json


class SAR2OptUNet(UNet2DConditionModel):

    def forward(self, sample, timestep, encoder_hidden_states, timestep_cond, cross_attention_kwargs,
                added_cond_kwargs):
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if added_cond_kwargs is not None:
            if 'sar' in added_cond_kwargs:
                image_embs = added_cond_kwargs.get("image_embeds")
                aug_emb = self.add_embedding(image_embs)
            else:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

        emb = emb + aug_emb if aug_emb is not None else emb
        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)

        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=None,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=None,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=None,
            )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=None,
                    encoder_attention_mask=None,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

class SAREncoder(nn.Module):
    def __init__(self,in_channels,ngf=50):
        super(SAREncoder, self).__init__()
        self.ngf = ngf
        self.encoder = nn.Sequential(
            # Encoder 1
            nn.Conv2d(in_channels=in_channels, out_channels=self.ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ngf),
            nn.LeakyReLU(0.2, inplace=True),

            # Encoder 2
            nn.Conv2d(in_channels=self.ngf, out_channels=self.ngf * 2, kernel_size=3, stride=2, padding=1),# half
            nn.BatchNorm2d(self.ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Encoder 3
            nn.Conv2d(in_channels=self.ngf * 2, out_channels=self.ngf * 4, kernel_size=3, stride=2, padding=1),# half 
            nn.BatchNorm2d(self.ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Encoder 4
            nn.Conv2d(in_channels=self.ngf * 4, out_channels=self.ngf * 5, kernel_size=3, stride=2, padding=1),# half 
            nn.BatchNorm2d(self.ngf * 5),
            nn.LeakyReLU(0.2, inplace=True),

        )

    def forward(self, x):
        bz = x.shape[0]
        out = self.encoder(x).reshape(bz, -1, 1280)
        return out


class SAR2OptUNetv2(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        in_channels = 1
        self.ngf = 2
        self.sar_encoder = nn.Sequential(
            # Encoder 1
            nn.Conv2d(in_channels=in_channels, out_channels=self.ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ngf),
            nn.LeakyReLU(0.2, inplace=True),

            # Encoder 2
            nn.Conv2d(in_channels=self.ngf, out_channels=self.ngf * 2, kernel_size=3, stride=2, padding=1),# half
            nn.BatchNorm2d(self.ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Encoder 3
            nn.Conv2d(in_channels=self.ngf * 2, out_channels=self.ngf * 4, kernel_size=3, stride=2, padding=1),# half 
            nn.BatchNorm2d(self.ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Encoder 4
            nn.Conv2d(in_channels=self.ngf * 4, out_channels=self.ngf * 5, kernel_size=3, stride=2, padding=1),# half 
            nn.BatchNorm2d(self.ngf * 5),
            nn.LeakyReLU(0.2, inplace=True),

        )

    def forward(self, sample, timestep, sar_image=None,
                encoder_hidden_states=None, 
                timestep_cond=None, cross_attention_kwargs=None,
                added_cond_kwargs=None):
        
        if encoder_hidden_states is None:
            assert sar_image is not None
            bz = sample.shape[0]
            encoder_hidden_states = self.sar_encoder(sar_image).reshape(bz, -1, 1280)

        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if added_cond_kwargs is not None:
            if 'sar' in added_cond_kwargs:
                image_embs = added_cond_kwargs.get("image_embeds")
                aug_emb = self.add_embedding(image_embs)
            else:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

        emb = emb + aug_emb if aug_emb is not None else emb
        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)

        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=None,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=None,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=None,
            )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=None,
                    encoder_attention_mask=None,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample



class SAR2OptUNetv3(UNet2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        
    def forward(self, sample, timestep):
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        return sample





# 3*64*64
if __name__ == '__main__':
    model = SAR2OptUNetv2(
            sample_size=256,
            in_channels=3,
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
    model.to("cuda")
    opt_image = torch.randn(8, 3, 256, 256).to("cuda")
    sar_image = torch.randn(8, 1, 256, 256).to("cuda")

    timestep = torch.tensor(1.0)
    re = model(opt_image, timestep, sar_image , None, None, None)
    print(re.shape)