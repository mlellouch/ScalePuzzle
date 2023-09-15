from __future__ import annotations

import random

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
import operator
import parse

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

    def __init__(self, piece_index: np.ndarray, piece_id, number_of_pieces=1):
        self.piece_mask = piece_index
        self.piece_exists = (self.piece_mask > 0).astype(np.uint8)
        self.number_of_pieces = number_of_pieces
        self.neighbor_mask = self._calculate_neighbor_convolution_mask()
        self.piece_id = piece_id

    def split(self):
        split_pieces = []
        for piece_index_number in self.piece_mask[self.piece_mask != 0]:
            split_pieces.append(JoinedPieces(
                piece_index=np.array([[0,0,0],[0, piece_index_number, 0], [0,0,0]], dtype=int),
                piece_id=piece_index_number -1,
                number_of_pieces=1
            ))

        return split_pieces


    def get_piece_frame(self, patches_data:single_image_dataset.ImagePatchesDataset):
        patch_size = patches_data.patch_size
        all_locations = np.argwhere(self.piece_mask != 0)
        ys = (all_locations[:, 0].min(), all_locations[:, 0].max())
        xs = (all_locations[:, 1].min(), all_locations[:, 1].max())

        y_frame_size = (ys[1] - ys[0] + 1) * patch_size
        x_frame_size = (xs[1] - xs[0] + 1) * patch_size
        frame = np.zeros(shape=[y_frame_size, x_frame_size, 3], dtype=np.uint8)

        for y in range(ys[0], ys[1]+1):
            for x in range(xs[0], xs[1]+1):
                current_patch = self.piece_mask[y,x]
                if current_patch == 0:
                    continue
                y_frame_start = (y - ys[0]) * patch_size
                x_frame_start = (x - xs[0]) * patch_size
                patch_to_copy = patches_data[current_patch - 1]
                frame[y_frame_start:y_frame_start+patch_size, x_frame_start:x_frame_start+patch_size, :] = patch_to_copy

        return frame

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
        return [JoinedPieces(piece_index=np.array([[0,0,0],[0, idx+1, 0], [0,0,0]], dtype=int), piece_id=idx, number_of_pieces=1) for idx in range(len(self.patches_data))]

    def calculate_pair_ratings(self):

        cache_dir = Path('./cache')
        dir_file_name = cache_dir / Path(f'{self.image.parts[-1].replace(".jpg", "").replace(".png", "")}')
        dir_file_name.mkdir(exist_ok=True, parents=True)
        pairs_file = dir_file_name / Path('pairs.npy')
        if pairs_file.exists():
            return np.load(str(pairs_file))

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

        np.save(str(pairs_file), pair_ratings)
        return pair_ratings

    def __init__(self, pair_matcher: PairMatcher, scale_validator: ScaleValidator, image: Path):
        self.pair_matcher = pair_matcher
        self.scale_validator = scale_validator
        self.image = image

        self.initial_threshold = 0.98
        self.threshold = self.initial_threshold
        self.split_threshold = 0.0

        self.patches_data = single_image_dataset.ImagePatchesDataset(image_path=image)
        self.pair_ratings = None
        self.pieces = self.init_pieces()

    @staticmethod
    def crop_piece_mask(piece_mask):
        locs = np.argwhere(piece_mask > 0)
        min_x, max_x = locs[:, 1].min(), locs[:, 1].max()
        min_y, max_y = locs[:, 0].min(), locs[:, 0].max()

        cropped = piece_mask[min_y: max_y+1, min_x:max_x+1]

        out = np.pad(cropped, [(1, 1 if cropped.shape[0] % 2 == 1 else 2), (1, 1 if cropped.shape[1] % 2 == 1 else 2)])
        return out

    def check_piece(self, new_piece, n=16):
        piece_mask = new_piece[0]
        for i in range(piece_mask.shape[0]):
            for j in range(piece_mask.shape[1]):
                if piece_mask[i,j] != 0:
                    if (piece_mask[i,j+1] != 0) and (piece_mask[i,j] + 1 != piece_mask[i, j+1]):
                        return False
                    if (piece_mask[i+1,j] != 0) and (piece_mask[i,j] + n != piece_mask[i+1, j]):
                        return False
        return True

    def should_join_pieces(self, p1, p2):
        for new_piece in p1.get_all_possible_joins(p2):
            if self.rate_join(new_piece) > self.threshold:
                if not self.check_piece(new_piece, n=16):
                    print(new_piece[0])
                piece_mask, boolean_mask, number_of_pairs = new_piece
                new_piece_mask = self.crop_piece_mask(piece_mask=piece_mask)
                new_final_piece = JoinedPieces(new_piece_mask, piece_id=min(p1.piece_id, p2.piece_id),number_of_pieces=p1.number_of_pieces + p2.number_of_pieces)
                return True, new_final_piece

        return False, None

    def should_join_pieces_optimal(self, p1, p2):
        max_rating = 0.0
        max_piece = None
        for new_piece in p1.get_all_possible_joins(p2):
            current_rating = self.rate_join(new_piece)
            if current_rating > max_rating:
                max_rating = current_rating
                piece_mask, boolean_mask, number_of_pairs = new_piece
                new_piece_mask = self.crop_piece_mask(piece_mask=piece_mask)
                new_final_piece = JoinedPieces(new_piece_mask, piece_id=min(p1.piece_id, p2.piece_id),number_of_pieces=p1.number_of_pieces + p2.number_of_pieces)
                max_piece = new_final_piece

        return max_rating, max_piece

    def should_join_pieces_optimal_ver2(self, p1, p2):
        max_rating = 0.0
        max_piece = None
        for new_piece in p1.get_all_possible_joins(p2):
            current_rating = self.rate_join_ver2(new_piece)
            if current_rating > max_rating:
                max_rating = current_rating
                piece_mask, boolean_mask, number_of_pairs = new_piece
                new_piece_mask = self.crop_piece_mask(piece_mask=piece_mask)
                new_final_piece = JoinedPieces(new_piece_mask, piece_id=min(p1.piece_id, p2.piece_id),number_of_pieces=p1.number_of_pieces + p2.number_of_pieces)
                max_piece = new_final_piece

        return max_rating, max_piece

    def solve_puzzle_greedy(self, animation_file='test.mp4'):
        self.pair_ratings = self.calculate_pair_ratings()
        random.shuffle(self.pieces) # truly test our algo

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(animation_file, fourcc, 1.0, (1024, 1024))

        progress_bar = tqdm(desc='solving puzzle', total=len(self.init_pieces()))
        self.load_state()
        pieces = self.pieces
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

            else:
                out.write(cv2.cvtColor(self.get_animation_frame(pieces, 1024, number_of_pieces=5), cv2.COLOR_BGR2RGB))

            if self.threshold < self.split_threshold:
                random.shuffle(pieces)
                piece_to_split = pieces.pop(0)
                split = piece_to_split.split()
                pieces += split
                self.threshold = self.initial_threshold

        out.release()
        progress_bar.close()
        self.save_state()
        print(pieces[0].piece_mask)

    def solve_puzzle_optimal(self, animation_file='test_optimal.mp4'):
        self.pair_ratings = self.calculate_pair_ratings()
        random.shuffle(self.pieces)  # truly test our algo

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(animation_file, fourcc, 1.0, (1024, 1024))

        pieces = self.pieces
        progress_bar = tqdm(desc='solving puzzle', total=len(self.init_pieces()))
        while len(pieces) > 1:
            max_rating = 0
            best_pair = (0, 1)
            for i in range(len(pieces)):
                for j in range(i + 1, len(pieces)):
                    rating, new_piece = self.should_join_pieces_optimal(pieces[i], pieces[j])
                    if rating > max_rating:
                        max_rating = rating
                        best_pair = (i,j)

            rating, new_piece = self.should_join_pieces_optimal(pieces[best_pair[0]], pieces[best_pair[1]])
            pieces.pop(best_pair[1])
            pieces.pop(best_pair[0])
            pieces.append(new_piece)
            progress_bar.update(1)
            out.write(cv2.cvtColor(self.get_animation_frame(pieces, 1024, number_of_pieces=5), cv2.COLOR_BGR2RGB))


        out.release()
        progress_bar.close()

        print(pieces[0].piece_mask)


    def solve_puzzle_optimal_ver2(self, animation_file='test_optimal_ver2.mp4'):
        self.pair_ratings = self.calculate_pair_ratings()
        random.shuffle(self.pieces)  # truly test our algo

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(animation_file, fourcc, 1.0, (1024, 1024))

        pieces = self.pieces
        progress_bar = tqdm(desc='solving puzzle', total=len(self.init_pieces()))
        while len(pieces) > 1:
            max_rating = 0
            best_pair = (0, 1)
            for i in range(len(pieces)):
                for j in range(i + 1, len(pieces)):
                    rating, new_piece = self.should_join_pieces_optimal_ver2(pieces[i], pieces[j])
                    if rating > max_rating:
                        max_rating = rating
                        best_pair = (i,j)

            rating, new_piece = self.should_join_pieces_optimal_ver2(pieces[best_pair[0]], pieces[best_pair[1]])
            pieces.pop(best_pair[1])
            pieces.pop(best_pair[0])
            pieces.append(new_piece)
            progress_bar.update(1)
            out.write(cv2.cvtColor(self.get_animation_frame(pieces, 1024, number_of_pieces=5), cv2.COLOR_BGR2RGB))


        out.release()
        progress_bar.close()

        print(pieces[0].piece_mask)

    def _get_average_pair_ratings(self, new_piece_mask, new_boolean_mask):
        all_ratings = []
        first_piece_locations = np.argwhere(new_boolean_mask == 1)
        number_of_connected_pieces = 0
        for y,x in first_piece_locations:
            # since piece masks are starting from 1, subtract 1 to each
            is_piece_connected = False
            if new_boolean_mask[y-1, x] == 2:
                is_piece_connected = True
                all_ratings.append(self.pair_ratings[new_piece_mask[y,x] - 1, new_piece_mask[y-1, x] - 1, 3])
            if new_boolean_mask[y+1, x] == 2:
                is_piece_connected = True
                all_ratings.append(self.pair_ratings[new_piece_mask[y,x] - 1, new_piece_mask[y+1, x] - 1, 4])
            if new_boolean_mask[y, x-1] == 2:
                is_piece_connected = True
                all_ratings.append(self.pair_ratings[new_piece_mask[y,x] - 1, new_piece_mask[y, x-1] - 1, 1])
            if new_boolean_mask[y, x+1] == 2:
                is_piece_connected = True
                all_ratings.append(self.pair_ratings[new_piece_mask[y,x] - 1, new_piece_mask[y, x+1] - 1, 2])

            if is_piece_connected:
                number_of_connected_pieces += 1

        return sum(all_ratings) / len(all_ratings), len(all_ratings), number_of_connected_pieces

    def rate_join(self, new_piece):
        piece_mask, boolean_mask, number_of_pairs = new_piece
        average_pair_rating, number_of_pairs, number_of_connected_pieces = self._get_average_pair_ratings(new_piece_mask=piece_mask, new_boolean_mask=boolean_mask)

        # first method, ignores scale validator
        # using a sigmoind-esque functon, s.t. f(1) = 1, and converges to ~ 1.3, so multiply it by some scale
        # this function is used by the number of pairs

        scale = 1.5
        value = ((1 / (1 + (np.e ** -number_of_pairs))) * (1 + (np.e ** -1))) ** scale
        return average_pair_rating * value

    def rate_join_ver2(self, new_piece):
        piece_mask, boolean_mask, number_of_pairs = new_piece
        average_pair_rating, number_of_pairs, number_of_connected_pieces = self._get_average_pair_ratings(new_piece_mask=piece_mask, new_boolean_mask=boolean_mask)

        # first method, ignores scale validator
        # using a sigmoind-esque functon, s.t. f(1) = 1, and converges to ~ 1.3, so multiply it by some scale
        # this function is used by the number of pairs

        scale = 1.5
        value = (number_of_pairs / number_of_connected_pieces) * ((1 / (1 + (np.e ** -number_of_pairs))) * (1 + (np.e ** -1))) ** scale
        return average_pair_rating * value

    def get_animation_frame(self, pieces: List[JoinedPieces], frame_size, number_of_pieces: int):
        pieces = sorted(pieces, key=lambda i: (i.number_of_pieces, i.piece_id), reverse=True)
        image_list = [p.get_piece_frame(self.patches_data) for p in pieces]
        frame_height = 0  # Initialize to 0
        frame_width = 0  # Initialize to 0
        images_to_add = []
        for img in image_list:
            h, w, channel = img.shape

            new_height = max(frame_height, h)
            new_width = frame_width + w

            if new_height >= frame_size or new_width >= frame_size:
                break

            frame_height = new_height
            frame_width += new_width
            images_to_add.append(img)

        spacing = 32  # You can adjust the spacing as needed

        # Create the larger frame with white background
        # frame = np.zeros((frame_height, frame_width + (len(image_list) - 1) * spacing), dtype=np.uint8)
        frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)

        # Initialize the starting position for each image
        x_start = 0

        # Place each image within the frame with spacing
        for img in images_to_add:
            h, w, channel = img.shape
            frame[spacing:h + spacing, x_start:x_start + w] = img
            x_start += w + spacing

        # Now 'frame' contains all the images placed within it with spacing

        return frame

    def get_last_state(self):
        cache_dir = Path('./cache')
        dir_file_name = cache_dir / Path(f'{self.image.parts[-1].replace(".jpg", "").replace(".png", "")}')
        dir_file_name.mkdir(exist_ok=True, parents=True)
        max_state = -1
        for file in dir_file_name.iterdir():
            try:
                state_number = parse.parse('{:d}.npz', file.parts[-1])[0]
                max_state = max(max_state, state_number)
            except:
                continue
        return max_state

    def save_state(self):
        # check if exists in cache
        cache_dir = Path('./cache')
        dir_file_name = cache_dir / Path(f'{self.image.parts[-1].replace(".jpg", "").replace(".png", "")}')
        dir_file_name.mkdir(exist_ok=True, parents=True)
        last_state_path = dir_file_name / Path(f'{self.get_last_state() + 1}.npz')
        np.savez(last_state_path, [piece.piece_mask for piece in self.pieces])

    def load_state(self):
        cache_dir = Path('./cache')
        dir_file_name = cache_dir / Path(f'{self.image.parts[-1].replace(".jpg", "").replace(".png", "")}')
        dir_file_name.mkdir(exist_ok=True, parents=True)
        last_state_path = dir_file_name / Path(f'{self.get_last_state()}.npz')
        if last_state_path.exists():
            pieces = np.load(str(last_state_path))

    def get_validator_rating(self, indeces):
        patch_size = self.patches_data.patch_size
        image_to_test = np.zeros(shape=[indeces.shape[0] * patch_size, indeces.shape[1] * patch_size])
        for i in range(indeces.shape[0]):
            for j in range(indeces.shape[1]):
                image_to_test[i * patch_size: i * patch_size + patch_size, j * patch_size: j * patch_size + patch_size] = \
                    self.patches_data[indeces[i,j]]

        self.scale_validator.infer(image_to_test)

    def rate_scale(self, piece: JoinedPieces, scales=(2,3,4)):
        # at the moment our method doesn't accept hole (this could be changed)
        output = np.zeros_like(piece.piece_mask)
        count = np.ones_like(piece.piece_mask, dtype=np.float32) * 1e-6
        for scale in scales:
            kernel = np.ones((scale, scale), dtype=np.uint8)
            locations_that_can_be_tested = correlate2d(piece.piece_exists, kernel, mode='valid') == (scales ** 2)
            for location in np.argwhere(locations_that_can_be_tested):
                count[location[0], location[1]] += 1
                output[location[0], location[1]] += self.get_validator_rating(
                    piece.piece_mask[location[0]: location[0] + scale, location[1]: location[1] + scale]
                )

        return output / count


    def solve_puzzle_optimal_with_scale(self, animation_file='test_optimal_scale.mp4'):
        self.pair_ratings = self.calculate_pair_ratings()
        self.load_state()
        random.shuffle(self.pieces)  # truly test our algo

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(animation_file, fourcc, 1.0, (1024, 1024))
        self.load_state()
        pieces = self.pieces
        for i in range(5):
            progress_bar = tqdm(desc='solving puzzle', total=len(self.init_pieces()))

            # puzzle solving iteration
            while len(pieces) > 1:
                max_rating = 0
                best_pair = (0, 1)
                for i in range(len(pieces)):
                    for j in range(i + 1, len(pieces)):
                        rating, new_piece = self.should_join_pieces_optimal(pieces[i], pieces[j])
                        if rating > max_rating:
                            max_rating = rating
                            best_pair = (i,j)

                rating, new_piece = self.should_join_pieces_optimal(pieces[best_pair[0]], pieces[best_pair[1]])
                pieces.pop(best_pair[1])
                pieces.pop(best_pair[0])
                pieces.append(new_piece)
                progress_bar.update(1)
                out.write(cv2.cvtColor(self.get_animation_frame(pieces, 1024, number_of_pieces=5), cv2.COLOR_BGR2RGB))

            self.save_state()
            # puzzle breaking
            scale_rating = self.rate_scale(pieces[0])
            out.write(cv2.cvtColor(scale_rating, cv2.COLOR_GRAY2BGR))
            pieces_to_break_off = scale_rating < self.scale_threshold
            new_pieces = pieces[0].break_off(pieces_to_break_off)
            pieces = new_pieces

            progress_bar.close()

        out.release()
        print(pieces[0].piece_mask)

if __name__ == '__main__':
    pair_matcher = PairMatcher(model_log_path=Path('./models/fully_trained'))
    scale_validator = ScaleValidator(model_log_path=Path('./models/scale_validator'))
    image = Path('./data/images/test_large/nature.jpg')

    solver = Solver(pair_matcher=pair_matcher, scale_validator=scale_validator, image=image)
    solver.solve_puzzle_optimal_ver2(animation_file='test_optimal_ver3.mp4')

