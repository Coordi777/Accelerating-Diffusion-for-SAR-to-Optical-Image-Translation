import torch
from torch import nn
from diffusers import UNet2DConditionModel, UNet2DModel


class UNetAsDiscriminator(UNet2DConditionModel):
    def __init__(self, uncond=False, bucket_dataset=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Delete any other components not needed.
        modules_to_keep = ["conv_in", "down_blocks", "mid_block", "time_proj", "time_embedding", "add_embedding", "add_time_proj"]
        modules_to_drop = [name for name, _ in self.named_children() if name not in modules_to_keep]
        for name in modules_to_drop:
            delattr(self, name)

        # Add prediction head
        self.uncond = uncond
        if bucket_dataset or self.sample_size == 64:
            self.head = nn.Sequential(
                nn.Conv2d(1280 if uncond else 1280*2, 1280, 3, 1, 1), # 32
                nn.GroupNorm(32, 1280),
                nn.SiLU(True),
                nn.Conv2d(1280, 1280, 4, 2, 1), # 16
                nn.GroupNorm(32, 1280),
                nn.SiLU(True),
                nn.Conv2d(1280, 1280, 4, 2, 1), # 8
                nn.GroupNorm(32, 1280),
                nn.SiLU(True),
                nn.Conv2d(1280, 1280, 4, 2, 1), # 4
                nn.GroupNorm(32, 1280),
                nn.SiLU(True),
                nn.AdaptiveAvgPool2d((1, 1)), # 1
                # nn.Conv2d(1280, 1280, 4, 2, 0),  # 1
                # nn.GroupNorm(32, 1280),
                # nn.SiLU(True),
                nn.Flatten(),
                nn.Linear(1280, 1)
            )
        else:
            self.head = nn.Sequential(
                nn.Conv2d(1280 if uncond else 1280 * 2, 1280, 3, 1, 1),  # 32
                nn.GroupNorm(32, 1280),
                nn.SiLU(True),
                nn.Conv2d(1280, 1280, 4, 2, 1),  # 16
                nn.GroupNorm(32, 1280),
                nn.SiLU(True),
                nn.Conv2d(1280, 1280, 4, 2, 1),  # 8
                nn.GroupNorm(32, 1280),
                nn.SiLU(True),
                nn.Conv2d(1280, 1280, 4, 2, 1),  # 4
                nn.GroupNorm(32, 1280),
                nn.SiLU(True),
                # nn.AdaptiveAvgPool2d((1, 1)),  # 1
                nn.Conv2d(1280, 1280, 4, 2, 0),  # 1
                nn.GroupNorm(32, 1280),
                nn.SiLU(True),
                nn.Flatten(),
                nn.Linear(1280, 1)
            )

    def get_feature(self, sample, timestep, encoder_hidden_states, added_cond_kwargs):
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        emb = self.time_proj(timesteps).type_as(sample)
        emb = self.time_embedding(emb)

        if self.config.addition_embed_type == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
            emb = emb + aug_emb

        # 2. pre-process
        sample = self.conv_in(sample)

         # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if getattr(downsample_block, "has_cross_attention", False):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        return sample

    def forward(self, sample1, timestep1, sample2, timestep2, encoder_hidden_states, added_cond_kwargs=None, sample3=None, timestep3=None):
        feature1 = self.get_feature(sample1, timestep1, encoder_hidden_states, added_cond_kwargs) if not self.uncond else None
        feature2 = self.get_feature(sample2, timestep2, encoder_hidden_states, added_cond_kwargs)
        feature12 = torch.cat([feature1, feature2], dim=1) if feature1 is not None else feature2
        logits12 = self.head(feature12)

        if sample3 is not None:
            feature3 = self.get_feature(sample3, timestep3, encoder_hidden_states, added_cond_kwargs)
            feature13 = torch.cat([feature1, feature3], dim=1) if feature1 is not None else feature3
            logits13 = self.head(feature13)
            return logits12, logits13

        return logits12


class SAR2OptUNetv_as_Dis(UNet2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        super().__init__(*args, **kwargs)

        # Delete any other components not needed.
        modules_to_keep = ["conv_in", "down_blocks", "mid_block", "time_proj", "time_embedding", "add_embedding", "add_time_proj"]
        modules_to_drop = [name for name, _ in self.named_children() if name not in modules_to_keep]
        for name in modules_to_drop:
            delattr(self, name)

        self.head = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1), # 8
            nn.GroupNorm(32, 1024),
            nn.SiLU(True),
            nn.Conv2d(1024, 1024, 4, 2, 1), # 4
            nn.GroupNorm(32, 1024),
            nn.SiLU(True),
            nn.AdaptiveAvgPool2d((1, 1)), # 1
            nn.Flatten(),
            nn.Linear(1024, 1)
        )


    def get_feature(self, sample, timestep):
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

        return sample
        
    def forward(self, sample1=None, timestep1=None, sample2=None, timestep2=None, sample3=None, timestep3=None):
        feature1 = self.get_feature(sample1, timestep1)
        feature2 = self.get_feature(sample2, timestep2)
        feature12 = torch.cat([feature1, feature2], dim=1) if feature2 is not None else feature1
        logits12 = self.head(feature12)

        if sample3 is not None:
            feature3 = self.get_feature(sample3, timestep3)
            feature13 = torch.cat([feature1, feature3], dim=1) if feature1 is not None else feature3
            logits13 = self.head(feature13)
            return logits12, logits13

        return logits12
        

        return logits12





if __name__ == "__main__":
    model = SAR2OptUNetv_as_Dis(
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
            ))
    x1 = torch.randn([1, 4, 256, 256])
    t1 = torch.randint(0, 1000, [1])

    x2 = torch.randn([1, 4, 256, 256])
    t2 = torch.randint(0, 1000, [1])

    o1,o2 = model(x1,t1, x2, t2,x2, t2)
    print(o1.shape)