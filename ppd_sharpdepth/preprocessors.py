from abc import ABC, abstractmethod
from PIL import Image
from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor
import torch
from torchvision.transforms.functional import pil_to_tensor

from ppd_sharpdepth.ppd.utils.transform import resize_keep_aspect

float_dtypes = [torch.float16, torch.float32, torch.float64, torch.bfloat16]

class PreProcessor(ABC):
    @staticmethod
    @abstractmethod
    def run(rgb_int_1chw: torch.Tensor, device: torch.device, float_dtype: torch.dtype):
        pass

class MarigoldPreProcessor(PreProcessor):
    @staticmethod
    def run(rgb_int_1chw: torch.Tensor, device: torch.device, float_dtype: torch.dtype):
        assert rgb_int_1chw.dtype == torch.int32, f"rgb_int_1chw must be of dtype torch.int32, got {rgb_int_1chw.dtype}"
        assert float_dtype in float_dtypes, f"float_dtype must be one of {float_dtypes}, got {float_dtype}"

        input_size = rgb_int_1chw.shape
        assert (
            4 == rgb_int_1chw.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        image_processor = MarigoldImageProcessor(vae_scale_factor=8, do_normalize=False)

        # NOTE: These values should ideally be set as params. If required, need to refactor class. For now, these vals do not seem to change so we are fine.
        processing_res = 0
        resample_method = "bilinear"

        rgb_float_1chw_resized, padding, original_resolution = image_processor.preprocess(rgb_int_1chw, processing_res, resample_method, device)  # [N,3,PPH,PPW]
        
        assert rgb_float_1chw_resized.dtype == float_dtype, f"rgb_float_1chw_resized must be of dtype {float_dtype}, got {rgb_float_1chw_resized.dtype}"
        return rgb_float_1chw_resized, padding, original_resolution

class PixelPerfectDepthPreProcessor(PreProcessor):
    @staticmethod
    def run(rgb_int_1chw: torch.Tensor, device: torch.device, float_dtype: torch.dtype):
        assert rgb_int_1chw.dtype == torch.int32, f"rgb_int_1chw must be of dtype torch.int32, got {rgb_int_1chw.dtype}"
        assert float_dtype in float_dtypes, f"float_dtype must be one of {float_dtypes}, got {float_dtype}"

        rgb_float_1chw = rgb_int_1chw.to(float_dtype) / 255.0
        rgb_float_hwc = rgb_float_1chw.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        resized_rgb_float_HpWpC = resize_keep_aspect(rgb_float_hwc)
        hp, wp, _ = resized_rgb_float_HpWpC.shape
        rgb_float_1chw_resized = torch.from_numpy(resized_rgb_float_HpWpC).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=rgb_float_1chw.dtype)

        assert rgb_float_1chw_resized.dtype == float_dtype, f"rgb_float_1chw_resized must be of dtype {float_dtype}, got {rgb_float_1chw_resized.dtype}"
        return rgb_float_1chw_resized, None, None

