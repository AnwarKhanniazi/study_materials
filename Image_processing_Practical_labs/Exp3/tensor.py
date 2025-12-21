from PIL import Image
from torchvision import transforms

# Load image (RGB)
img = Image.open("pk.jpg").convert("RGB")

# Convert to tensor
transform = transforms.ToTensor()
tensor_img = transform(img)

print("Tensor shape:", tensor_img.shape)   # [C, H, W]
print("Tensor dtype:", tensor_img.dtype)
