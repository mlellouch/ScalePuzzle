from __future__ import annotations
import numpy as np
from train_pair import PairMatcher
from train_validator import ScaleValidator
from typing import List
from pathlib import Path
from data import tiled_pair_dataset
from data import single_image_dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import cv2
from scipy.signal import convolve2d, correlate2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

neighbor_kernel = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)

class JoinedPieces:
    piece_mask: np.ndarray

    def _calculate_neighbor_convolution_mask(self):
        piece_exists = (self.piece_mask > 0).astype(np.uint8)
        neighbors = cv2.dilate(piece_exists, neighbor_kernel).astype(int)
        neighbors[piece_exists > 0] = -100
        return neighbors

    def __init__(self, piece_index: np.ndarray, number_of_pieces=1):
        self.piece_mask = piece_index
        self.piece_exists = (self.piece_mask > 0).astype(np.uint8)
        self.number_of_pieces = number_of_pieces
        self.neighbor_mask = self._calculate_neighbor_convolution_mask()

    def get_all_possible_joins(self, new_piece: JoinedPieces):
        """
        return all the new possible piece masks, where this current piece is touching the new one.
        Also for one return a mask that says from which piece is each point from
        :param new_piece:
        :return:
        """

        # this for bug fixing, atm not the prettiest solution but whateves
        y_add = new_piece.piece_exists.shape[0] // 2
        x_add = new_piece.piece_exists.shape[1] // 2

        self_neighbor_mask = np.pad(self.neighbor_mask, [(y_add, y_add), (x_add, x_add)])
        self_piece_mask = np.pad(self.piece_mask, [(y_add, y_add), (x_add, x_add)])
        # now normal code

        possible_locations = correlate2d(self_neighbor_mask, new_piece.piece_exists, mode='full')
        help_kernel = np.zeros_like(new_piece.piece_exists)
        help_kernel[help_kernel.shape[0] // 2, help_kernel.shape[1] // 2] = 1
        help_for_future = correlate2d(self_piece_mask, help_kernel, mode='full')

        possible_indeces = np.argwhere(possible_locations > 0)

        for location in possible_indeces:
            other_piece_size = np.array(new_piece.piece_exists.shape)
            start = location - (other_piece_size // 2)
            end = location + (other_piece_size // 2) + 1

            # add the new piece
            possible_join = help_for_future.copy()
            join_mask = possible_join.copy()
            join_mask[help_for_future != 0] = 1
            possible_join[start[0]: end[0], start[1]: end[1]][new_piece.piece_mask != 0] = new_piece.piece_mask[new_piece.piece_mask != 0]
            join_mask[start[0]: end[0], start[1]: end[1]][new_piece.piece_mask != 0] = 2

            number_of_pairs = possible_locations[location[0], location[1]]
            yield possible_join, join_mask, number_of_pairs


class Solver:

    def init_pieces(self):
        return [JoinedPieces(piece_index=np.array([[0,0,0],[0, idx+1, 0], [0,0,0]], dtype=int), number_of_pieces=1) for idx in range(len(self.patches_data))]

    def calculate_pair_ratings(self):
        d = tiled_pair_dataset.SingleImageDataset(self.image)
        loader = DataLoader(d, batch_size=16)
        pair_ratings = np.zeros(shape=[len(d.pieces), len(d.pieces), 5])
        for pairs in tqdm(loader, desc='Calculating all pair matches'):
            (center_index, tiled_index), tiled_image = pairs
            center_index = center_index.numpy()
            tiled_index = tiled_index.numpy()
            tiled_image = tiled_image.to(device=device)
            current_pair_ratings = self.pair_matcher.infer(tiled_image).cpu().numpy()
            for center_idx, tile_idx, ratings in zip(center_index, tiled_index, current_pair_ratings):
                pair_ratings[center_idx, tile_idx, :] = ratings

        return pair_ratings

    def __init__(self, pair_matcher: PairMatcher, scale_validator: ScaleValidator, image: Path):
        self.pair_matcher = pair_matcher
        self.scale_validator = scale_validator
        self.image = image
        self.threshold = 0.9

        self.patches_data = single_image_dataset.ImagePatchesDataset(image_path=image)
        self.pair_ratings = None

    @staticmethod
    def crop_piece_mask(piece_mask):
        locs = np.argwhere(piece_mask > 0)
        min_x, max_x = locs[:, 1].min(), locs[:, 1].max()
        min_y, max_y = locs[:, 0].min(), locs[:, 0].max()

        cropped = piece_mask[min_y: max_y+1, min_x:max_x+1]

        out = np.pad(cropped, [(1, 1 if cropped.shape[0] % 2 == 1 else 2), (1, 1 if cropped.shape[1] % 2 == 1 else 2)])
        return out

    def should_join_pieces(self, p1, p2):
        for new_piece in p1.get_all_possible_joins(p2):
            if self.rate_join(new_piece) > self.threshold:
                piece_mask, boolean_mask, number_of_pairs = new_piece
                new_piece_mask = self.crop_piece_mask(piece_mask=piece_mask)
                new_final_piece = JoinedPieces(new_piece_mask, number_of_pieces=p1.number_of_pieces + p2.number_of_pieces)
                return True, new_final_piece

        return False, None

    def solve_puzzle(self):
        self.pair_ratings = self.calculate_pair_ratings()
        pieces = self.init_pieces()

        progress_bar = tqdm(desc='solving puzzle', total=len(self.init_pieces()))
        while len(pieces) > 1:
            join_occured = False
            for i in range(len(pieces)):
                for j in range(i+1, len(pieces)):
                    was_joined, new_piece = self.should_join_pieces(pieces[i], pieces[j])
                    if was_joined:
                        pieces.pop(j)
                        pieces.pop(i)
                        pieces.append(new_piece)
                        progress_bar.update(1)
                        join_occured = True
                        break
                if join_occured:
                    break

            if not join_occured:
                print(f'No piece joining occurred. changing threshold {self.threshold} -> {self.threshold * 0.9}')
                self.threshold *= 0.9

        progress_bar.close()
        print(pieces[0].piece_mask)

    def _get_average_pair_ratings(self, new_piece_mask, new_boolean_mask):
        all_ratings = []
        first_piece_locations = np.argwhere(new_boolean_mask == 1)
        for y,x in first_piece_locations:
            # since piece masks are starting from 1, subtract 1 to each
            if new_boolean_mask[y-1, x] == 2:
                all_ratings.append(self.pair_ratings[new_piece_mask[y,x] - 1, new_piece_mask[y-1, x] - 1, 1])
            if new_boolean_mask[y+1, x] == 2:
                all_ratings.append(self.pair_ratings[new_piece_mask[y,x] - 1, new_piece_mask[y+1, x] - 1, 2])
            if new_boolean_mask[y, x-1] == 2:
                all_ratings.append(self.pair_ratings[new_piece_mask[y,x] - 1, new_piece_mask[y, x-1] - 1, 3])
            if new_boolean_mask[y, x+1] == 2:
                all_ratings.append(self.pair_ratings[new_piece_mask[y,x] - 1, new_piece_mask[y, x+1] - 1, 4])

        return sum(all_ratings) / len(all_ratings), len(all_ratings)

    def rate_join(self, new_piece):
        piece_mask, boolean_mask, number_of_pairs = new_piece
        average_pair_rating, number_of_pairs = self._get_average_pair_ratings(new_piece_mask=piece_mask, new_boolean_mask=boolean_mask)

        # first method, ignores scale validator
        # using a sigmoind-esque functon, s.t. f(1) = 1, and converges to ~ 1.3, so multiply it by some scale
        # this function is used by the number of pairs

        scale = 1.5
        value = ((1 / (1 + (np.e ** -number_of_pairs))) * (1 + (np.e ** -1))) ** scale
        return average_pair_rating * value

if __name__ == '__main__':
    pair_matcher = PairMatcher(model_log_path=Path('./models/first_test'))
    scale_validator = None
    image = Path('./data/images/test/3_small.jpg')

    solver = Solver(pair_matcher=pair_matcher, scale_validator=scale_validator, image=image)
    solver.solve_puzzle()

