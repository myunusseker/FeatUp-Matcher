import torch
from PIL import Image
from torchvision import transforms

# Set your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load FeatUp model (it expects an image as input, not DINOv2 features)
upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True).to(device)
upsampler.eval()

# Load and preprocess the image
image_path = "sample-images/lid.jpg"
image = Image.open(image_path).convert("RGB")

# Resize the image to what FeatUp expects (likely 224x224)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_tensor = preprocess(image).unsqueeze(0).to(device)

# Apply FeatUp to the raw image directly
with torch.no_grad():
    upsampled_features = upsampler(input_tensor)

# Print the output shape
print("Upsampled features shape:", upsampled_features.shape)
