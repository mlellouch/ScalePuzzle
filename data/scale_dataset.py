import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
import random
from typing import List
import matplotlib.pyplot as plt


class ScaleDataset(Dataset):

    def __init__(self, images_path: Path, patch_size=32, number_of_possible_patches=(2, 3, 4, 5)):
        self.images_path = images_path
        self.all_images = list(self.images_path.iterdir())
        self.patch_size = patch_size
        self.number_of_possible_patches = number_of_possible_patches
        self.transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.all_images[idx]))
        number_of_patches = random.choice(self.number_of_possible_patches)

        first_chosen_y = np.random.randint(low=0, high=(image.shape[0] // self.patch_size) - number_of_patches + 1) * self.patch_size
        first_chosen_x = np.random.randint(low=0, high=(image.shape[1] // self.patch_size) - number_of_patches + 1) * self.patch_size

        second_chosen_y = np.random.randint(low=0, high=(image.shape[0] // self.patch_size) - number_of_patches + 1) * self.patch_size
        second_chosen_x = np.random.randint(low=0, high=(image.shape[1] // self.patch_size) - number_of_patches + 1) * self.patch_size

        while first_chosen_x == second_chosen_x and first_chosen_y == second_chosen_y:
            second_chosen_y = np.random.randint(low=0, high=(image.shape[0] // self.patch_size) - number_of_patches + 1) * self.patch_size
            second_chosen_x = np.random.randint(low=0, high=(image.shape[1] // self.patch_size) - number_of_patches + 1) * self.patch_size

        first_correct = image[first_chosen_y: first_chosen_y + (number_of_patches * self.patch_size), first_chosen_x: first_chosen_x + (number_of_patches * self.patch_size)]
        second_correct = image[second_chosen_y: second_chosen_y + (number_of_patches * self.patch_size),second_chosen_x: second_chosen_x + (number_of_patches * self.patch_size)]

        patches = np.random.randint(low=0, high=2, size=[number_of_patches, number_of_patches], dtype=bool)
        while patches.all() and (not patches.any()):
            patches = np.random.randint(low=0, high=2, size=[number_of_patches, number_of_patches], dtype=bool)
        full_patches = np.repeat(np.repeat(patches, repeats=self.patch_size, axis=0), repeats=self.patch_size, axis=1) # multiply patches
        full_patches = np.stack([full_patches, full_patches, full_patches], axis=2)

        first_combined = np.where(full_patches, first_correct, second_correct)
        second_combined = np.where(~full_patches, first_correct, second_correct)

        labels = [0, 0, 1, 1]
        return np.stack([first_correct, second_correct, first_combined, second_combined], axis=0), labels

if __name__ == '__main__':
    d = ScaleDataset(images_path=Path('./images/train'))

    a = d[1]
    a = 0











