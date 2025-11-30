if __name__ == "__main__":

    from huggingface_hub import hf_hub_download
    import torch
    from ppd_sharpdepth.ppd.models.ppd import PixelPerfectDepth
    import sys

    device = torch.device('cuda')

    if sys.argv[-1] == "push":
        print("Pushing Pixel Perfect Depth to Hugging Face Hub")

        ckpt_path = hf_hub_download(repo_id="gangweix/pixel-perfect-depth", filename="ppd.pth")
        semantics_path = hf_hub_download(repo_id="depth-anything/Depth-Anything-V2-Large", filename="depth_anything_v2_vitl.pth")

        model = PixelPerfectDepth(semantics_pth=semantics_path, sampling_steps=4, depth_anything_v2_encoder="vitl", depth_anything_v2_features=256, depth_anything_v2_out_channels=[256, 512, 1024, 1024])
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        model = model.to(device).eval()
        model.requires_grad_(False)

        model.push_to_hub("andrew-healey/sharpdepth", subfolder="ppd")
    elif sys.argv[-1] == "pull":
        print("Pulling Pixel Perfect Depth from Hugging Face Hub")
        model = PixelPerfectDepth.from_pretrained("andrew-healey/sharpdepth", subfolder="ppd")
        model = PixelPerfectDepth.from_pretrained("andrew-healey/pixel-perfect-depth")
        print("Pulled!")
    else:
        raise ValueError(f"Invalid command: {sys.argv[-1]}")