from mmengine.config import Config

from estimator.models.patchrefiner import PatchRefiner

from collections import OrderedDict
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

config_path = "./configs/patchrefiner_zoedepth/pr_u4k.py"
checkpoint_fine_path = "./work_dir/zoedepth/u4k/pr/checkpoint_36.pth"
checkpoint_coarse_path = "./work_dir/zoedepth/u4k/coarse_pretrain/checkpoint_24.pth"
image_shape = (480, 640)
patch_split = (2, 2)

# Load config file
cfg = Config.fromfile(config_path)

# Build the model
model = PatchRefiner(cfg.model.config)
print("Model instantiated!")

# Load coarse branch checkpoint first
coarse_ckpt = torch.load(checkpoint_coarse_path, map_location='cpu')
model.coarse_branch.load_state_dict(coarse_ckpt, strict=True)
print("Coarse branch loaded!")

# Load fine branch checkpoint
fine_ckpt = torch.load(checkpoint_fine_path)
model.load_state_dict(fine_ckpt["model_state_dict"], strict=False)
print("Fine branch loaded!")

# Change to eval mode
model.eval()
print("Switched to eval mode!")

# Load and preprocess image
img = Image.open("input_image.jpg").convert("RGB")
img = F.interpolate(img, size=image_shape, mode="bilinear", align_corners=False)
rgb_13hw = transforms.ToTensor()(img).unsqueeze(0)
print("Preprocessed rgb images!")

# Load and preprocess coarse depth map
depth = np.load("input_depth.npy")
depth_11hw = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
print("Preprocessed depth maps!")

# Split coarse depth into patches
B, C, H, W = depth_tensor.shape
patch_h = H // patch_split[0]
patch_w = W // patch_split[1]

crop_depths = []
for i in range(patch_split[0]):
    for j in range(patch_split[1]):
        patch = depth_11hw[:, :, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
        crop_depths.append(patch)
crop_depths_p1hw = torch.cat(crop_depths, dim=0)
print("Split coarse depth into patches!")

# Move model and input to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
rgb_13hw = rgb_13hw.to(device)

with torch.no_grad():
    refined_depth = model(image_hr=rgb_13hw, crop_depths=crop_depths_p1hw)

refined_depth_np_hw = refined_depth.squeeze(0).cpu().numpy()
np.save("./result.npy", refined_depth_np_hw)
print("Refined depth map complete!")

plt.imshow(refined_depth_np, cmap='plasma')
plt.colorbar()
plt.savefig("./result.rgb", bbox_inches='tight', pad_inches=0)
plt.close()
print("Done!")