import sys
import numpy as np
import pickle
from numpy.linalg import norm
import matplotlib
import time

matplotlib.use('TkAgg')
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from featup.util import pca

class Dinov2Matcher:

    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", smaller_edge_size=448,
                 patch_size=14, device="cuda", ref_img_name='water_bottle.jpeg', ref_patch=(2, 16)):
        self.repo_name = repo_name
        self.model_name = model_name
        self.smaller_edge_size = smaller_edge_size
        self.patch_size = patch_size
        self.device = device

        self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            transforms.ToTensor(),
        ])

        self.ref_img_name = ref_img_name
        self.ref_patch = ref_patch

        # Prepare reference image
        self.ref_img = Image.open(ref_img_name).convert("RGB")

        # Extract features for the reference image
        self.ref_img_tensor, self.ref_grid, self.ref_scale = self.prepare_image(self.ref_img)
        self.ref_features = self.extract_features(self.ref_img_tensor).reshape(*self.ref_grid, -1)
        self.ref_norm = norm(self.ref_features[self.ref_patch])

        # Show the reference image and reference patch
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(self.ref_img_tensor.squeeze().permute(1, 2, 0))
        rect = patches.Rectangle(
            (ref_patch[1] * self.patch_size, ref_patch[0] * self.patch_size), self.patch_size, self.patch_size,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        plt.title('Reference Image with Reference Patch')
        plt.show()
        plt.imshow(self.feature_rgb)
        plt.show()
        upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).to(self.device)
        #upsampler.model = self.model
        print(self.ref_img_tensor.unsqueeze(0).to(device).shape)
        upsampled_features = upsampler(self.ref_img_tensor.unsqueeze(0).to(device))

    def prepare_image(self, rgb_image_numpy):
        image = rgb_image_numpy
        image_tensor = self.transform(image)
        resize_scale = image.width / image_tensor.shape[2]

        height, width = image_tensor.shape[1:]  # C x H x W
        cropped_width, cropped_height = width - width % self.patch_size, height - height % self.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.patch_size, cropped_width // self.patch_size)
        return image_tensor, grid_size, resize_scale

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            feature_tensor = self.model.get_intermediate_layers(image_batch)[0]
            tokens = feature_tensor.squeeze()
        [feature_tensor_rgb], _ = pca([feature_tensor.reshape(*self.ref_grid, -1).unsqueeze(0).permute(0, 3, 1, 2)], use_torch_pca=False)
        print(feature_tensor_rgb.shape)
        self.feature_rgb = feature_tensor_rgb[0].permute(1, 2, 0).cpu().numpy()
        return tokens.cpu().numpy()

def main():
    dm = Dinov2Matcher(repo_name='facebookresearch/dinov2', model_name='dinov2_vitl14', patch_size=14, ref_img_name='sample-images/view4.png', ref_patch=(27, 17))


if __name__ == "__main__":
    main()