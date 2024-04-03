from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomImageLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self