from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
from ppd_sharpdepth.ppd.utils.timesteps import Timesteps
from ppd_sharpdepth.ppd.utils.schedule import LinearSchedule
from ppd_sharpdepth.ppd.utils.sampler import EulerSampler
from ppd_sharpdepth.ppd.utils.transform import image2tensor, resize_1024, resize_1024_crop, resize_keep_aspect

from ppd_sharpdepth.ppd.models.depth_anything_v2.dpt import DepthAnythingV2
from ppd_sharpdepth.ppd.models.dit import DiT

from huggingface_hub import PyTorchModelHubMixin
from diffusers import ConfigMixin, ModelMixin

from typing import List, Any, Dict, Union, Optional

class PixelPerfectDepth(ModelMixin, ConfigMixin):
    config_name = "config.json"

    def __init__(
        self,
        semantics_pth:Optional[str]=None,
        sampling_steps:int=4,
        depth_anything_v2_encoder:str='vitl',
        depth_anything_v2_features:int=256,
        depth_anything_v2_out_channels:List[int]=[256, 512, 1024, 1024],
        dit_in_channels:int=4,
    ):
        super(PixelPerfectDepth, self).__init__()

        self.semantics_encoder = DepthAnythingV2(
            encoder=depth_anything_v2_encoder,
            features=depth_anything_v2_features,
            out_channels=depth_anything_v2_out_channels
        )

        if semantics_pth is not None:
            self.semantics_encoder.load_state_dict(torch.load(semantics_pth, map_location='cpu'), strict=False)
        self.semantics_encoder = self.semantics_encoder.eval()
        self.dit = DiT(in_channels=dit_in_channels)

        self.sampling_steps = sampling_steps

        self.schedule = LinearSchedule(T=1000)
        self.sampling_timesteps = Timesteps(
            T=self.schedule.T,
            steps=self.sampling_steps,
        )
        self.sampler = EulerSampler(
            schedule=self.schedule,
            timesteps=self.sampling_timesteps,
            prediction_type='velocity'
        )

        # required for ConfigMixin
        self.register_to_config(
            sampling_steps=sampling_steps,
            depth_anything_v2_encoder=depth_anything_v2_encoder,
            depth_anything_v2_features=depth_anything_v2_features,
            depth_anything_v2_out_channels=depth_anything_v2_out_channels,
            dit_in_channels=dit_in_channels
        )
    
    @torch.no_grad()
    def infer_image(self, image_bgr_hwc, use_fp16: bool = True):
        # Resize the image to match the training resolution area while keeping the original aspect ratio.
        resize_image_bgr_hwc = resize_keep_aspect(image_bgr_hwc)
        image_rgb_1chw = image2tensor(resize_image_bgr_hwc)
        image_rgb_1chw = image_rgb_1chw.to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
            depth = self.forward_test(image_rgb_1chw)
        return depth, resize_image_bgr_hwc
    
    @torch.no_grad()
    def forward_test(self, image_rgb_1chw):

        semantics = self.semantics_prompt(image_rgb_1chw)
        cond = image_rgb_1chw - 0.5
        latent = torch.randn(size=[cond.shape[0], 1, cond.shape[2], cond.shape[3]]).to(self.device)
        
        for timestep in self.sampling_timesteps:
            input = torch.cat([latent, cond], dim=1)
            pred = self.dit(x=input, semantics=semantics, timestep=timestep)
            latent = self.sampler.step(pred=pred, x_t=latent, t=timestep)

        return latent + 0.5


    @torch.no_grad()
    def semantics_prompt(self, image_rgb_hwc):
        with torch.no_grad():
            semantics = self.semantics_encoder(image_rgb_hwc)
        return semantics
    
    # we define forward() to just be .dit() since .dit is the only trainable part of the network
    def forward(self, *args, **kwargs):
        return self.dit(*args, **kwargs)