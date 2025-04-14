import sys
import numpy as np
import pickle
from numpy.linalg import norm
import matplotlib
import time
from scipy.ndimage import shift

matplotlib.use('TkAgg')
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
from featup.util import norm as Tnorm
from featup.util import unnorm as Tunnorm
from featup.plotting import plot_feats, plot_lang_heatmaps
from hubconf import UpsampledBackbone


def find_highest_heatmap_point(heatmap_img):
    # Convert heatmap image to grayscale in order to find the whitest point
    gray_heatmap = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2GRAY)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_heatmap)
    max_loc_flatten = heatmap_img.shape[0] * max_loc[0] + max_loc[1]
    return max_loc, max_val, max_loc_flatten

def prepare_image(image, patch_size=14):
    transform = T.Compose([
        T.Resize(size=448, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        Tnorm,
    ])
    image_tensor = transform(image)

    height, width = image_tensor.shape[1:]  # C x H x W
    cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]

    return image_tensor


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinoo = UpsampledBackbone(model_name="dinov2-l", use_norm=False).to(device)

    patch_size = 14
    #reference_patch = (27, 13)
    reference_patch = (4, 17)
    reference_pixel = (reference_patch[0]*patch_size+7, reference_patch[1]*patch_size+7)
    reference_image = Image.open('sample-images/handmixer.jpeg').convert("RGB")
    target_image = Image.open('sample-images/mixer.jpg').convert("RGB")

    reference_tensor = prepare_image(reference_image, patch_size=patch_size).unsqueeze(0).to(device)
    target_tensor = prepare_image(target_image, patch_size=patch_size).unsqueeze(0).to(device)

    # Show the reference image and target image side-by-side

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(Tunnorm(reference_tensor)[0].permute(1, 2, 0).cpu().numpy())
    rect = patches.Rectangle(
        (reference_patch[1] * patch_size, reference_patch[0] * patch_size),
        patch_size, patch_size,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax1.scatter(reference_pixel[1], reference_pixel[0])
    ax1.add_patch(rect)
    ax1.set_title('Reference Image with Reference Patch')
    ax1.axis('off')  # Hide axis for better visualization

    # Display target image
    ax2.imshow(Tunnorm(target_tensor)[0].permute(1, 2, 0).cpu().numpy())
    ax2.set_title('Target Image')
    ax2.axis('off')  # Hide axis for better visualization

    plt.show()

    #upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=False).to(device)
    # Calculate Reference Features and Normalize
    reference_hr_feats = dinoo(reference_tensor)
    reference_hr_feats = F.interpolate(reference_hr_feats, size=(reference_tensor.shape[3], reference_tensor.shape[2]), mode='bilinear', align_corners=False)
    reference_lr_feats = dinoo.model(reference_tensor)
    plot_feats(Tunnorm(reference_tensor)[0], reference_lr_feats[0], reference_hr_feats[0])

    reference_lr_feats = reference_lr_feats[0].permute(1, 2, 0).cpu().detach().numpy()
    reference_lr_feats /= norm(reference_lr_feats, axis=2, keepdims=True)

    reference_hr_feats = reference_hr_feats[0].permute(1, 2, 0).cpu().detach().numpy()
    reference_hr_feats /= norm(reference_hr_feats, axis=2, keepdims=True)

    # Calculate Target Features and Normalize
    target_hr_feats = dinoo(target_tensor)
    target_hr_feats = F.interpolate(target_hr_feats, size=(target_tensor.shape[3], target_tensor.shape[2]), mode='bilinear', align_corners=False)
    target_lr_feats = dinoo.model(target_tensor)
    plot_feats(Tunnorm(target_tensor)[0], target_lr_feats[0], target_hr_feats[0])

    target_lr_feats = target_lr_feats[0].permute(1, 2, 0).cpu().detach().numpy()
    target_lr_feats /= norm(target_lr_feats, axis=2, keepdims=True)

    target_hr_feats = target_hr_feats[0].permute(1, 2, 0).cpu().detach().numpy()
    target_hr_feats /= norm(target_hr_feats, axis=2, keepdims=True)

    # Calculate Similarities for the reference patch
    ref_patch_feat_lr = reference_lr_feats[reference_patch[0], reference_patch[1]]  # shape (C,)
    cosine_map_lr = np.nan_to_num(target_lr_feats @ ref_patch_feat_lr)
    cosine_map_lr = (cosine_map_lr - np.min(cosine_map_lr)) / (np.max(cosine_map_lr) - np.min(cosine_map_lr))

    ref_patch_feat_hr = reference_hr_feats[reference_pixel[0], reference_pixel[1]]  # shape (C,)
    cosine_map_hr = (np.nan_to_num(target_hr_feats @ ref_patch_feat_hr) + 1) / 2.  # shape (H, W)
    cosine_map_hr = (cosine_map_hr - np.min(cosine_map_hr)) / (np.max(cosine_map_hr) - np.min(cosine_map_hr))

    # Convert heatmap to image using OpenCV
    heat_map_lr = cv2.applyColorMap((cosine_map_lr * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    heat_map_lr = cv2.cvtColor(heat_map_lr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    heat_map_lr = cv2.resize(heat_map_lr, (target_tensor.shape[3], target_tensor.shape[2]), interpolation=cv2.INTER_NEAREST)
    heat_map_hr = cv2.applyColorMap((cosine_map_hr * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    heat_map_hr = cv2.cvtColor(heat_map_hr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    target_image = target_tensor[0].permute(1, 2, 0).cpu().detach().numpy()

    target_heat_map_lr_combined = cv2.addWeighted(target_image, 0.1, heat_map_lr, 0.9, 0)
    target_heat_map_hr_combined = cv2.addWeighted(target_image, 0.1, heat_map_hr, 0.9, 0)

    # Create a figure with two subplots arranged in a single row
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Display the low-resolution heatmap image on the left
    ax1.imshow(target_heat_map_lr_combined)
    ax1.set_title('Low-Resolution Heatmap Image')
    ax1.axis('off')  # Hide axis for better visualization

    # Display the high-resolution heatmap image on the right
    ax2.imshow(target_heat_map_hr_combined)
    ax2.set_title('High-Resolution Heatmap Image')
    ax2.axis('off')  # Hide axis for better visualization

    # Display the side-by-side images
    plt.show()
