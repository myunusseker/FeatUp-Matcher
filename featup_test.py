import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import torch
import torchvision.transforms as T
from PIL import Image

from featup.util import norm, unnorm
from featup.plotting import plot_feats, plot_lang_heatmaps
available_models = torch.hub.list("mhamilton723/FeatUp")
print(available_models)

def prepare_image(image):
    patch_size = 14
    transform = T.Compose([
        T.Resize(size=448, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        norm,
    ])
    image_tensor = transform(image)

    height, width = image_tensor.shape[1:]  # C x H x W
    cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]

    return image_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_norm = True
upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(device)
upsampler.eval()
upsampler.model.eval()
image_path = "sample-images/view4.png"
image = Image.open(image_path).convert("RGB")
print(image.size)
image_tensor = prepare_image(image).unsqueeze(0).to(device)
print('Input Shape', image_tensor.shape)
hr_feats = upsampler(image_tensor)
lr_feats = upsampler.model(image_tensor)
print(hr_feats.size())
print(lr_feats.size())
plot_feats(unnorm(image_tensor)[0], lr_feats[0], hr_feats[0])