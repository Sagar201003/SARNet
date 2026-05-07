import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

def preprocess_image(image):
    """
    Preprocess PIL image for the model.
    Grayscale SAR image -> [1, H, W] tensor in [-1, 1]
    """
    # Convert to grayscale
    image = image.convert("L")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Add batch dimension
    tensor = transform(image).unsqueeze(0)
    return tensor

def postprocess_tensor(tensor):
    """
    Convert [-1, 1] tensor back to PIL RGB image.
    """
    # Remove batch dimension and move to CPU
    tensor = tensor.squeeze(0).cpu().detach()
    
    # Denormalize [-1, 1] to [0, 1]
    tensor = (tensor + 1.0) / 2.0
    
    # Clamp to [0, 1]
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy array [0, 255]
    image_np = tensor.numpy().transpose(1, 2, 0)
    image_np = (image_np * 255).astype(np.uint8)
    
    # Convert back to PIL Image
    return Image.fromarray(image_np)
