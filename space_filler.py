import multiprocessing as mp
import random
import time
from multiprocessing import shared_memory

import numpy as np

# Constants
MAP_WIDTH = 5
MAP_HEIGHT = 5
PIECE_SIZE = 4
NUM_WORKERS = 4
OVERLAP = PIECE_SIZE - 1  # Overlap size


pieces = [
    np.array([[0, 1, 0], [1, 1, 1]]),
    np.array([[1, 1, 1], [0, 1, 0]]),
    np.array([[1, 1], [1, 1]]),
    np.array([[1, 1, 0], [0, 1, 1]]),
]

piece_dict = {i: piece for i, piece in enumerate(pieces)}


def create_tetris_pieces(num_pieces=4, pieces=pieces):
    return random.choices(pieces, k=num_pieces)


pieces = create_tetris_pieces(num_pieces=2, pieces=pieces)
print(pieces)


def rotate_piece(piece, rotations=1):
    piece = np.rot90(piece, k=3)
    return piece


piece = pieces[0]
rotated_piece = rotate_piece(piece)
print(rotated_piece)


def create_map(width, height, empty=True):
    if empty:
        tetris_map = np.zeros((width, height))

    else:
        probabilities = [0.8, 0.2]
        tetris_map = np.random.choice([0, 1], (width, height), p=probabilities)

    shm = shared_memory.SharedMemory(create=True, size=tetris_map.nbytes)
    np_array_shared = np.ndarray(tetris_map.shape, dtype=np.int8, buffer=shm.buf)
    np_array_shared[:] = tetris_map[:]
    return np_array_shared, shm


tetris_map, shm = create_map(MAP_WIDTH, MAP_HEIGHT, empty=False)
print(tetris_map)


def evaluate_board(board):
    holes = np.sum(np.diff(board, axis=0) < 0)
    complete_lines = np.sum(np.all(board, axis=1))
    return -holes + 10 * complete_lines


def can_place_piece(piece, local_map, x, y):
    return np.all(
        (local_map[x : x + piece.shape[0], y : y + piece.shape[1]] + piece) <= 1
    )


def available_rotations(piece, local_map, x, y):
    rotation_image = piece.copy()
    available_rotations = []
    for i in range(4):
        rotation_image = rotate_piece(rotation_image)
        if can_place_piece(rotation_image, local_map, x, y):
            available_rotations.append(i)
    return available_rotations


can_be_put = can_place_piece(piece, tetris_map, 0, 0)
print(can_be_put)
print(evaluate_board(tetris_map))
print(available_rotations(piece, tetris_map, 0, 0))
shm.close()  # close shared memory, so it doesn't get deleted but we are not using it anymore
shm.unlink()  # delete shared memory
