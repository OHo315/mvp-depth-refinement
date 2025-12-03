from abc import ABC, abstractmethod
from PIL import Image
from diffusers.pipelines.marigold.marigold_image_processing import MarigoldImageProcessor
import torch

class PreProcessor(ABC):
    @staticmethod
    @abstractmethod
    def run(input_image: Image.Image | torch.Tensor, device: torch.device, dtype: torch.dtype):
        pass

class MarigoldPreProcessor(PreProcessor):
    @staticmethod
    def run(input_image: Image.Image | torch.Tensor, device: torch.device, dtype: torch.dtype):
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb_int_1chw = pil_to_tensor(input_image)
            rgb_int_1chw = rgb_int_1chw.unsqueeze(0)  # [1, rgb, H, W], dtype int
        elif isinstance(input_image, torch.Tensor):
            rgb_int_1chw = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb_int_1chw.shape
        assert (
            4 == rgb_int_1chw.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        image_processor = MarigoldImageProcessor(vae_scale_factor=8, do_normalize=False)

        # NOTE: These values should ideally be set as params. If required, need to refactor class. For now, these vals do not seem to change so we are fine.
        processing_res = 0
        resample_method = "bilinear"

        image, padding, original_resolution = image_processor.preprocess(rgb_int_1chw, processing_res, resample_method, device)  # [N,3,PPH,PPW]
        
        return image, padding, original_resolution, rgb_int_1chw

class PixelPerfectDepthPreProcessor(PreProcessor):
    @staticmethod
    def run(input_image: Image.Image | torch.Tensor, device: torch.device, dtype: torch.dtype):
        image, padding, original_resolution, rgb_int_1chw = MarigoldPreProcessor.run(input_image, device, dtype)
        rgb_float_1chw = rgb_int_1chw.to(dtype) / 255.0
        rgb_float_hwc = rgb_float_1chw.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        resized_rgb_float_HpWpC = resize_keep_aspect(rgb_float_hwc)
        hp, wp, _ = resized_rgb_float_HpWpC.shape
        rgb_float_1chw_resized = torch.from_numpy(resized_rgb_float_HpWpC).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=rgb_float_1chw.dtype)
        rgb_float_1chw = rgb_float_1chw_resized

        return rgb_float_1chw_resized, padding, original_resolution, rgb_int_1chw

