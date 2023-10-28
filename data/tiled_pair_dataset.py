import random

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import List
import matplotlib.pyplot as plt
from scipy.signal import correlate2d


class TiledPairDataset(Dataset):

    def __init__(self, images_path: Path, patch_size=32):
        self.images_path = images_path
        self.all_images = list(self.images_path.iterdir())
        self.patch_size = patch_size
        self.transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.all_images)

    def _get_random_non_neighbors(self, image):
        center_y = np.random.randint(low=1, high=image.shape[0] // self.patch_size)
        center_x = np.random.randint(low=1, high=image.shape[1] // self.patch_size)

        non_neighbor_y = np.random.randint(low=0, high=image.shape[0] // self.patch_size)
        non_neighbor_x = np.random.randint(low=0, high=image.shape[1] // self.patch_size)

        while (abs(non_neighbor_x - center_x) <= 1) and (abs(non_neighbor_y - center_y) <= 1):
            non_neighbor_y = np.random.randint(low=0, high=image.shape[0] // self.patch_size)
            non_neighbor_x = np.random.randint(low=0, high=image.shape[1] // self.patch_size)

        center_patch, non_neighbor_patch = self.get_patch(image, (center_y, center_x)), \
                                           self.get_patch(image, (non_neighbor_y, non_neighbor_x))

        return self._get_tiled_image(center_patch=center_patch, tiled_patch=non_neighbor_patch)

    def _get_tiled_image(self, center_patch, tiled_patch):
        canvas = np.zeros(shape=[self.patch_size * 3, self.patch_size * 3, 3], dtype=center_patch.dtype)
        canvas[self.patch_size: self.patch_size * 2, self.patch_size: self.patch_size * 2] = center_patch

        canvas[:self.patch_size, self.patch_size: self.patch_size * 2] = tiled_patch
        canvas[self.patch_size*2:, self.patch_size: self.patch_size * 2] = tiled_patch
        canvas[self.patch_size: self.patch_size * 2, :self.patch_size] = tiled_patch
        canvas[self.patch_size: self.patch_size * 2, self.patch_size * 2:] = tiled_patch

        return canvas

    def get_patch(self, image, patch_coords):
        y, x = patch_coords
        return image[y * self.patch_size: y * self.patch_size + self.patch_size,
                       x * self.patch_size: x * self.patch_size + self.patch_size]

    def __getitem__(self, idx):
        image = np.array(Image.open(self.all_images[idx]))
        if image.shape[2] == 4:
            image = image[:, :, :3]
        chosen_y = np.random.randint(low=1, high=(image.shape[0] // self.patch_size) - 1)
        chosen_x = np.random.randint(low=1, high=(image.shape[1] // self.patch_size) - 1)

        center_patch = self.get_patch(image, (chosen_y, chosen_x))

        neighbors = {
            1: (chosen_y - 1, chosen_x),
            2: (chosen_y + 1, chosen_x),
            3: (chosen_y, chosen_x - 1),
            4: (chosen_y, chosen_x + 1)
        }

        outputs = []
        labels = []
        for label, second_patch in neighbors.items():
            labels.append(label)
            tiled_patch = self.get_patch(image, second_patch)
            outputs.append(self._get_tiled_image(center_patch=center_patch, tiled_patch=tiled_patch))

        for _ in range(4):
            labels.append(0)
            outputs.append(self._get_random_non_neighbors(image))

        final_images = [self.transforms(Image.fromarray(output_image)) for output_image in outputs]
        return torch.stack(final_images), torch.tensor(labels, dtype=torch.long)

class SandedPairDataset(TiledPairDataset):

    def __init__(self, images_path: Path, sand_factor=0.2):
        super().__init__(images_path, patch_size=32)
        self.sand_factor = sand_factor


    def find_contours(self, mask_image):
        kernel = np.array([
            [0, 0.25, 0],
            [0.25, 0, 0.25],
            [0, 0.25, 0]
        ])

        result = correlate2d(mask_image, kernel, mode='same', boundary='fill', fillvalue=0)
        return np.argwhere(result < 1.0)

    def sand_patch(self, full_patch):
        starting_pixels = self.patch_size ** 2
        mask = np.ones(shape=[self.patch_size, self.patch_size], dtype=float)

        for i in range(int(self.sand_factor * starting_pixels)):
            contours = self.find_contours(mask)
            pixel_to_remove = random.choice(contours)
            mask[pixel_to_remove[0], pixel_to_remove[1]] = 0

        sanded_patch = full_patch.copy()
        sanded_patch[mask == 0] = 0
        return sanded_patch

    def get_patch(self, image, patch_coords):
        y, x = patch_coords
        full_patch = image[y * self.patch_size: y * self.patch_size + self.patch_size,
                       x * self.patch_size: x * self.patch_size + self.patch_size]

        sanded_patch = self.sand_patch(full_patch)
        return sanded_patch

class SandedPairTestDataset(SandedPairDataset):

    def _choose_random_image(self):
        self.current_image = np.array(Image.open(random.choice(self.all_images)))

    def __init__(self, images_path: Path):
        super().__init__(images_path)
        self._choose_random_image()

    def __len__(self):
        # calculate the number of patches
        height_patches = self.current_image.shape[0] // self.patch_size
        width_patches = self.current_image.shape[1] // self.patch_size
        return height_patches * width_patches

    def __getitem__(self, idx):
        height_patches = self.current_image.shape[0] // self.patch_size
        width_patches = self.current_image.shape[1] // self.patch_size
        middle_y, middle_x = (height_patches // 2), (width_patches // 2)
        center_patch = self.get_patch(self.current_image, (middle_y, middle_x))

        current_y = idx // width_patches
        current_x = idx % width_patches

        second_patch = self.get_patch(self.current_image, (current_y, current_x))
        label = 0
        if current_y + 1 == middle_y and current_x == middle_x:
            label = 1
        if current_y - 1 == middle_y and current_x == middle_x:
            label = 2
        if current_y == middle_y and current_x + 1 == middle_x:
            label = 3
        if current_y == middle_y and current_x - 1 == middle_y:
            label = 4

        outputs = [self._get_tiled_image(center_patch=center_patch, tiled_patch=second_patch)]
        final_images = [self.transforms(Image.fromarray(output_image)) for output_image in outputs]
        return torch.stack(final_images), torch.tensor(label, dtype=torch.long)


import data.erosion as erosion
class ErodedPairDataset(TiledPairDataset):

    def __init__(self, images_path: Path, erosion_volume=20):
        super().__init__(images_path, patch_size=32)
        self.erosion_volume = erosion_volume


    def get_patch(self, image, patch_coords):
        y, x = patch_coords
        full_patch = image[y * self.patch_size: y * self.patch_size + self.patch_size,
                       x * self.patch_size: x * self.patch_size + self.patch_size]

        eroded_patch = erosion.erode_image(image=full_patch, noise_volume=self.erosion_volume)
        return eroded_patch

class ErodedPairTestDataset(ErodedPairDataset):

    def _choose_random_image(self):
        self.current_image = np.array(Image.open(random.choice(self.all_images)))
        self.chosen_y = np.random.randint(low=1, high=(self.current_image.shape[0] // self.patch_size) - 1)
        self.chosen_x = np.random.randint(low=1, high=(self.current_image.shape[1] // self.patch_size) - 1)

    def __init__(self, images_path: Path):
        super().__init__(images_path)
        self._choose_random_image()

    def __len__(self):
        # calculate the number of patches
        height_patches = self.current_image.shape[0] // self.patch_size
        width_patches = self.current_image.shape[1] // self.patch_size
        return height_patches * width_patches

    def __getitem__(self, idx):
        height_patches = self.current_image.shape[0] // self.patch_size
        width_patches = self.current_image.shape[1] // self.patch_size

        center_patch = self.get_patch(self.current_image, (self.chosen_y, self.chosen_x))

        current_y = idx // width_patches
        current_x = idx % width_patches

        second_patch = self.get_patch(self.current_image, (current_y, current_x))
        label = 0
        if current_y + 1 == self.chosen_y and current_x == self.chosen_x:
            label = 1
        if current_y - 1 == self.chosen_y and current_x == self.chosen_x:
            label = 2
        if current_y == self.chosen_y and current_x + 1 == self.chosen_x:
            label = 3
        if current_y == self.chosen_y and current_x - 1 == self.chosen_x:
            label = 4

        outputs = [self._get_tiled_image(center_patch=center_patch, tiled_patch=second_patch)]
        final_images = [self.transforms(Image.fromarray(output_image)) for output_image in outputs]
        return torch.stack(final_images), torch.tensor(label, dtype=torch.long)

class SingleImageDataset(TiledPairDataset):
    pieces: List[np.ndarray]

    def load_pieces(self):
        entire_image = np.array(Image.open(self.image_path))
        for y in range(0, entire_image.shape[0] // self.patch_size):
            for x in range(0, entire_image.shape[1] // self.patch_size):
                self.pieces.append(entire_image[y * self.patch_size:(y * self.patch_size) + self.patch_size,
                                   x * self.patch_size:(x * self.patch_size) + self.patch_size])

    def __init__(self, image_path: Path, patch_size=32):
        super().__init__(image_path.parent, patch_size)
        self.image_path = image_path
        self.pieces = []
        self.load_pieces()

    def __len__(self):
        return len(self.pieces) ** 2

    def __getitem__(self, idx):
        center_index = idx // len(self.pieces)
        tiled_index = idx % len(self.pieces)
        tiled_image = super()._get_tiled_image(self.pieces[center_index], self.pieces[tiled_index])
        return (center_index, tiled_index), self.transforms(Image.fromarray(tiled_image))


if __name__ == '__main__':
    # test
    d = SandedPairDataset(images_path=Path('./images/train'))

    while True:
        a = d[0]
