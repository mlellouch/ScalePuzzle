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
        return torch.stack([self.transforms(Image.fromarray(img)) for img in [first_correct, second_correct, first_combined, second_combined]]), \
               torch.tensor(labels, dtype=torch.long)

def get_zero_mask(patches: np.ndarray):
    start = patches.astype(np.uint8) + 1
    new_change = np.random.randint(low=0, high=2, size=[patches.shape[0], patches.shape[1]], dtype=np.uint8)
    while True:
        new_patch = start * new_change
        new_patch_is_good = True
        for i in [1,2]:
            if (start == i).sum() == 1:
                if (new_patch == i).sum() == 0:
                    new_patch_is_good = False

            if (start == i).sum() > 1:
                if (new_patch == i).sum() <= 1:
                    new_patch_is_good = False

        if new_patch_is_good:
            break
        else:
            new_change = np.random.randint(low=0, high=2, size=[patches.shape[0], patches.shape[1]], dtype=np.uint8)

    return new_change



class ScaleDatasetWithHoles(ScaleDataset):

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

        combined_patches = np.random.randint(low=0, high=2, size=[number_of_patches, number_of_patches], dtype=bool)
        while combined_patches.all() and (not combined_patches.any()):
            combined_patches = np.random.randint(low=0, high=2, size=[number_of_patches, number_of_patches], dtype=bool)

        patches_zero_mask = [get_zero_mask(combined_patches), get_zero_mask(~combined_patches),
                                      get_zero_mask(np.ones_like(combined_patches, dtype=bool)),
                                      get_zero_mask(np.zeros_like(combined_patches, dtype=bool))
                                      ]


        full_patches = np.repeat(np.repeat(combined_patches, repeats=self.patch_size, axis=0), repeats=self.patch_size, axis=1) # multiply patches
        full_patches = np.stack([full_patches, full_patches, full_patches], axis=2)

        full_zero_patches = []
        for zero_patch in patches_zero_mask:
            p = np.repeat(np.repeat(zero_patch, repeats=self.patch_size, axis=0), repeats=self.patch_size,
                      axis=1)  # multiply patches
            p = np.stack([p, p, p], axis=2)
            full_zero_patches.append(p)

        first_combined = np.where(full_patches, first_correct, second_correct)
        second_combined = np.where(~full_patches, first_correct, second_correct)

        outputs = [first_correct, second_correct, first_combined, second_combined]
        outputs = [z * o for z, o in zip(full_zero_patches, outputs)]

        labels = [0, 0, 1, 1]
        return torch.stack([self.transforms(Image.fromarray(img)) for img in outputs]), \
               torch.tensor(labels, dtype=torch.long)


if __name__ == '__main__':
    d = ScaleDataset(images_path=Path('./images/train'))

    a = d[1]
    a = 0











