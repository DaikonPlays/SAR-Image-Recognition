import torch
from PIL import Image
from torchvision import transforms
import torchvision

MSTAR_dir = '/Users/kevinyan/Downloads/MSTAR_TargetData/2S1/HB20013.JPG';  
image = Image.open(MSTAR_dir)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor()
])
x = transform(image)

# x = torch.randn(1, 1, 128, 128)
print(x.shape)
unfolded_x = x.unfold(2, 32, 32)
print(unfolded_x.shape)
unfolded_x = unfolded_x.unfold(3, 32, 32)
print(unfolded_x.shape)

unfolded_x = unfolded_x.contiguous().view(1, 1, 16, 32 * 32)
print(unfolded_x.shape)
unfolded_x = unfolded_x.view(-1, 32 * 32)
print(unfolded_x.shape)
