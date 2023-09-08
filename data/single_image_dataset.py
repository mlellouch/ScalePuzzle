import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import List
import matplotlib.pyplot as plt

class ImagePatchesDataset:
    pieces: torch.tensor

    def load_pieces(self):
        entire_image = np.array(Image.open(self.image_path))
        pieces = []
        for y in range(0, entire_image.shape[0] // self.patch_size):
            for x in range(0, entire_image.shape[1] // self.patch_size):
                new_piece = entire_image[y * self.patch_size:(y * self.patch_size) + self.patch_size,
                                   x * self.patch_size:(x * self.patch_size) + self.patch_size]
                pieces.append(self.transforms(new_piece))

        self.pieces = torch.stack(pieces, dim=0)


    def __init__(self, image_path: Path, patch_size=32):
        self.patch_size =patch_size
        self.image_path = image_path
        self.pieces = None
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.load_pieces()

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        return self.pieces[idx]
