import multiprocessing
import pickle
import queue
import random
import sys
import time
from collections import namedtuple
from multiprocessing.managers import SyncManager

import numpy as np

Move = namedtuple("Move", ["piece", "rotation", "row", "col"])

# Constants
MAP_WIDTH = 7
MAP_HEIGHT = 6
PIECE_SIZE = 9
DEPTH_LIMIT = 8

pieces_list = [
    np.array([[0, 1, 0], [1, 1, 1]]),
    np.array([[1, 1, 1], [0, 1, 0]]),
    np.array([[1, 1], [1, 1]]),
    np.array([[1, 1, 0], [0, 1, 1]]),
]


class MyManager(SyncManager):
    pass


class Node:
    def __init__(self, board, score, pieces_left, move=None, parent=None):
        self.board = board
        self.score = score
        self.pieces_left = pieces_left
        self.move = move
        self.parent = parent

    def __lt__(self, other):
        return self.score > other.score

    def expand(self):
        children = []
        for i, piece_no in enumerate(self.pieces_left):
            piece = pieces_list[piece_no]
            possible_moves = get_possible_moves(piece, piece_no, self.board)
            for move in possible_moves:
                rotated_piece = rotate_piece(piece, move.rotation)
                new_board = place_piece(rotated_piece, self.board, move.row, move.col)
                new_score = evaluate_board(new_board)
                new_pieces_left = np.delete(self.pieces_left, i)
                new_node = Node(new_board, new_score, new_pieces_left, move, self)
                children.append(new_node)
        return children

    def get_attributes(self):
        return {
            "board": self.board,
            "score": self.score,
            "pieces_left": self.pieces_left,
            "move": self.move,
            "parent": self.parent,
        }

    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# class SharedNode:
#     def __init__(self, max_bytes=100000000):
#         self.score = multiprocessing.Value("d", float("-inf"))
#         self.node = multiprocessing.Array("c", max_bytes)
#         self.size = multiprocessing.Value("i", 0)

#     def update(self, node):
#         with self.score.get_lock(), self.size.get_lock():
#             if node.score > self.score.value:
#                 self.score.value = node.score
#                 pickled_node = pickle.dumps(node)
#                 if len(pickled_node) > len(self.node):
#                     print(
#                         f"Warning: Pickled node size ({len(pickled_node)}) exceeds buffer size ({len(self.node)})"
#                     )
#                     return
#                 self.size.value = len(pickled_node)
#                 self.node[: self.size.value] = pickled_node

#     def is_better(self, node):
#         with self.score.get_lock():
#             return node.score > self.score.value

#     def get_node(self):
#         print(self.score.value)
#         return pickle.loads(self.node.value)


MyManager.register(
    "PriorityQueue", queue.PriorityQueue
)  # Register a shared PriorityQueue
# MyManager.register("SharedNode", SharedNode)
MyManager.register("Node", Node)


def Manager():
    m = MyManager()
    m.start()
    return m


num_workers = multiprocessing.cpu_count()  # Get the number of available CPU cores
print(f"Using {num_workers} workers")


def create_tetris_piece_numbers(num_pieces=4, pieces=pieces_list):
    return np.random.randint(len(pieces), size=num_pieces)


def rotate_piece(piece, rotations=1):
    return np.rot90(piece, k=rotations)


def create_map(width, height, empty=True, zero_prob=1.0):
    if empty:
        return np.zeros((width, height))
    else:
        return np.random.choice([0, 1], (width, height), p=[zero_prob, 1 - zero_prob])


def evaluate_board(board):
    holes = np.sum(np.diff(board, axis=0) < 0)
    complete_lines = np.sum(np.all(board, axis=1))
    complete_columns = np.sum(np.all(board, axis=0))
    return -holes + 10 * complete_lines


def can_place_piece(piece, local_map, x, y):
    if (
        x + piece.shape[0] > local_map.shape[0]
        or y + piece.shape[1] > local_map.shape[1]
    ):
        return False
    return np.all(
        (local_map[x : x + piece.shape[0], y : y + piece.shape[1]] + piece) <= 1
    )


def place_piece(piece, local_map, x, y):
    new_map = local_map.copy()
    new_map[x : x + piece.shape[0], y : y + piece.shape[1]] += piece
    return new_map


def available_rotations(piece, local_map, x, y):
    return [
        i for i in range(4) if can_place_piece(rotate_piece(piece, i), local_map, x, y)
    ]


def get_possible_moves(piece, piece_key, tetris_map):
    possible_moves = []
    for i in range(tetris_map.shape[0] - piece.shape[0] + 1):
        for j in range(tetris_map.shape[1] - piece.shape[1] + 1):
            available_rotations_at_position = available_rotations(
                piece, tetris_map, i, j
            )
            possible_moves.extend(
                [Move(piece_key, k, i, j) for k in available_rotations_at_position]
            )
    return possible_moves


def run_search(
    node_heap,
    depth_limit,
    best_node,
    lock,
    queue_empty_flag,
    node_expand_counter,
    expand_counter_lock,
):
    process_id = multiprocessing.current_process().name
    print(f"{process_id} started")
    while True:
        node = None
        with lock:
            if not node_heap.empty():
                node = node_heap.get()
                if node_heap.empty():
                    queue_empty_flag.value = True
        if node is None:
            if queue_empty_flag.value:
                # print("Maybe in deadlock")
                time.sleep(0.1)  # Wait a bit before checking again
                continue
            else:
                break  # Exit if queue is empty and flag is False
        # print(
        #     f"{process_id} got a node from queue, score: {node.get_attributes()['score']}"
        # )

        # with lock:
        if node.get_attributes()["score"] > best_node.get_attributes()["score"]:
            best_node.set_attributes(**node.get_attributes())
            print(
                f"{process_id} updated best_node, new score: {best_node.get_attributes()['score']}"
            )
            print(f"best move board: {best_node.get_attributes()['board']} ")
        # if node.score > best_node.score:
        #     # best_node.__dict__.update(node.__dict__)
        #     best_node.board = node.board
        #     best_node.score = node.score
        #     best_node.pieces_left = node.pieces_left
        #     best_node.move = node.move
        #     best_node.parent = node.parent

        # print(f"{process_id} calculating move sequence")
        if len(get_move_sequence(node)) >= depth_limit:
            continue
        # print(f"{process_id} move sequence length: {len(get_move_sequence(node))}")

        children = node.expand()
        with expand_counter_lock:
            node_expand_counter.value += 1
            if node_expand_counter.value % 100 == 0:
                sys.stdout.write(f"\rTotal expanded nodes: {node_expand_counter.value}")
                sys.stdout.flush()
            if node_expand_counter.value >= 1000:
                print(f"\n{process_id} reached node limit")
                break
        # expanded_nodes += len(children)
        # sys.stdout.write(f"\r{process_id} expanded nodes: {expanded_nodes}")
        # sys.stdout.flush()
        # print(f"{process_id} expanded node, got {len(children)} children")
        with lock:
            for child in children:
                node_heap.put(child)
                # print(f"{process_id} put a child in queue")
            if queue_empty_flag.value:
                queue_empty_flag.value = False
        # print(f"{process_id} finished")
    print(f"{process_id} finished")


def get_move_sequence(node):
    moves = []
    node_attributes = node.get_attributes()
    # print(node_attributes)
    while node_attributes["parent"] is not None:
        node_attributes = node.get_attributes()
        moves.append(node_attributes["move"])
        node = node_attributes["parent"]
    return list(reversed(moves))


if __name__ == "__main__":
    manager = Manager()
    tetris_map = create_map(MAP_WIDTH, MAP_HEIGHT, empty=False)
    piece_numbers = create_tetris_piece_numbers(PIECE_SIZE)
    root_node = Node(tetris_map, evaluate_board(tetris_map), piece_numbers, None, None)
    node_heap = manager.PriorityQueue()
    queue_empty_flag = multiprocessing.Value("b", False)
    node_expand_counter = multiprocessing.Value("i", 0)

    lock = manager.Lock()
    expand_counter_lock = manager.Lock()
    node_heap.put(root_node)
    start_time = time.time()
    best_node = manager.Node(
        root_node.board,
        root_node.score,
        root_node.pieces_left,
        root_node.move,
        root_node.parent,
    )

    # Create and start worker processes
    workers = []
    for _ in range(12):
        p = multiprocessing.Process(
            target=run_search,
            args=(
                node_heap,
                DEPTH_LIMIT,
                best_node,
                lock,
                queue_empty_flag,
                node_expand_counter,
                expand_counter_lock,
            ),
        )
        workers.append(p)
        p.start()

    # Wait for all worker processes to finish
    for p in workers:
        p.join()

    end_time = time.time()

    # best_node_obj = best_node.get_node()
    moves = get_move_sequence(best_node)
    best_node_attributes = best_node.get_attributes()
    print(f"Best score: {best_node_attributes['score']}")
    print(f"Number of pieces placed: {len(moves)-1} out of {len(piece_numbers)}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("\nMove sequence:")
    print(moves)
    for move in moves[1:]:
        print(
            f"Piece {move.piece}, rotation {move.rotation}, row {move.row}, col {move.col}"
        )
    print("\nInitial board:")
    print(tetris_map)
    print("\nFinal board:")
    print(best_node_attributes["board"])
