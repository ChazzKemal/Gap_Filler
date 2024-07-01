import multiprocessing as mp
import random
import time
from collections import namedtuple

import numpy as np

# Constants
MAP_WIDTH = 5
MAP_HEIGHT = 5
NUM_PROCESSES = mp.cpu_count()
DEPTH_LIMIT = 20  # Increased depth limit
NUM_PIECES = 5

Move = namedtuple("Move", ["piece", "rotation", "row", "col"])


class Node:
    def __init__(self, board, score, pieces_left, move=None, parent=None):
        self.board = board
        self.score = score
        self.pieces_left = pieces_left
        self.move = move
        self.parent = parent
        self.children = []


def create_tetris_pieces(num_pieces):
    pieces = [
        np.array([[1, 1], [1, 1]]),  # Square
        np.array([[1, 1, 1, 1]]),  # Line
        np.array([[1, 1, 1], [0, 1, 0]]),  # T-shape
    ]
    return [random.choice(pieces) for _ in range(num_pieces)]


def rotate_piece(piece):
    return [piece, np.rot90(piece), np.rot90(piece, 2), np.rot90(piece, 3)]


def can_place_piece(board, piece, row, col):
    if row + piece.shape[0] > board.shape[0] or col + piece.shape[1] > board.shape[1]:
        return False
    return np.all(
        (board[row : row + piece.shape[0], col : col + piece.shape[1]] + piece) <= 1
    )


def place_piece(board, piece, row, col):
    new_board = board.copy()
    new_board[row : row + piece.shape[0], col : col + piece.shape[1]] += piece
    return new_board


def get_possible_moves(board, piece):
    moves = []
    for rotation, rotated_piece in enumerate(rotate_piece(piece)):
        for row in range(board.shape[0] - rotated_piece.shape[0] + 1):
            for col in range(board.shape[1] - rotated_piece.shape[1] + 1):
                if can_place_piece(board, rotated_piece, row, col):
                    moves.append(Move(piece, rotation, row, col))
    return moves


def aggregate_height(board):
    return np.sum(board.shape[0] - np.argmax(board[::-1] > 0, axis=0))


def complete_lines(board):
    return np.sum(np.all(board > 0, axis=1))


def holes(board):
    return np.sum(np.diff((board > 0)[::-1], axis=0) < 0)


def bumpiness(board):
    heights = board.shape[0] - np.argmax(board[::-1] > 0, axis=0)
    return np.sum(np.abs(np.diff(heights)))


def evaluate_board(board, pieces_placed):
    weights = {
        "aggregate_height": -0.510066,
        "complete_lines": 0.760666,
        "holes": -0.35663,
        "bumpiness": -0.184483,
        "pieces_placed": 0.5,  # New weight for pieces placed
    }

    score = (
        weights["aggregate_height"] * aggregate_height(board)
        + weights["complete_lines"] * complete_lines(board)
        + weights["holes"] * holes(board)
        + weights["bumpiness"] * bumpiness(board)
        + weights["pieces_placed"] * pieces_placed  # Reward for placing more pieces
    )

    return score


def explore_tree(root, depth_limit):
    stack = [(root, 0)]
    best_node = root

    while stack:
        node, depth = stack.pop()

        if depth >= depth_limit or not node.pieces_left:
            if node.score > best_node.score:
                best_node = node
            continue

        piece = node.pieces_left[0]
        remaining_pieces = node.pieces_left[1:]

        for move in get_possible_moves(node.board, piece):
            rotated_piece = rotate_piece(move.piece)[move.rotation]
            new_board = place_piece(node.board, rotated_piece, move.row, move.col)
            pieces_placed = NUM_PIECES - len(remaining_pieces)
            new_score = evaluate_board(new_board, pieces_placed)

            child = Node(new_board, new_score, remaining_pieces, move, node)
            node.children.append(child)
            stack.append((child, depth + 1))

            if child.score > best_node.score:
                best_node = child

    return best_node


def parallel_explore(args):
    root, depth_limit = args
    return explore_tree(root, depth_limit)


def run_parallel_search(initial_board, pieces, depth_limit):
    root = Node(initial_board, evaluate_board(initial_board, 0), pieces)

    # Generate first level of the tree
    first_piece = pieces[0]
    remaining_pieces = pieces[1:]
    initial_moves = get_possible_moves(initial_board, first_piece)

    # Create initial nodes for parallel exploration
    initial_nodes = []
    for move in initial_moves:
        rotated_piece = rotate_piece(move.piece)[move.rotation]
        new_board = place_piece(initial_board, rotated_piece, move.row, move.col)
        new_score = evaluate_board(new_board, 1)
        child = Node(new_board, new_score, remaining_pieces, move, root)
        root.children.append(child)
        initial_nodes.append(child)

    # Parallel exploration
    with mp.Pool(NUM_PROCESSES) as pool:
        best_nodes = pool.map(
            parallel_explore, [(node, depth_limit - 1) for node in initial_nodes]
        )

    # Find the overall best node
    best_node = max(best_nodes, key=lambda x: x.score)

    return best_node


def get_move_sequence(node):
    moves = []
    while node.parent is not None:
        moves.append(node.move)
        node = node.parent
    return list(reversed(moves))


def print_moves(moves):
    for i, move in enumerate(moves, 1):
        piece_type = (
            "Square"
            if move.piece.shape == (2, 2)
            else "Line" if move.piece.shape == (1, 4) else "T-shape"
        )
        print(
            f"Move {i}: Place {piece_type} at row {move.row}, column {move.col} with rotation {move.rotation}"
        )


def print_board(board):
    for row in board:
        print("".join(["#" if cell else "." for cell in row]))


if __name__ == "__main__":
    initial_board = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=int)
    pieces = create_tetris_pieces(NUM_PIECES)

    start_time = time.time()
    best_node = run_parallel_search(initial_board, pieces, DEPTH_LIMIT)
    end_time = time.time()

    moves = get_move_sequence(best_node)

    print(f"Best score: {best_node.score}")
    print(f"Number of pieces placed: {len(moves)} out of {NUM_PIECES}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("\nMove sequence:")
    print_moves(moves)
    print("\nFinal board:")
    print_board(best_node.board)
