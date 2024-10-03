import chess
import numpy as np

rank_map = {
    '1': '8', '2': '7', '3': '6', '4': '5',
    '5': '4', '6': '3', '7': '2', '8': '1'
}

def mirror_uci(move):
        move = [*move]
        move[1], move[3] = rank_map[move[1]], rank_map[move[3]]
        return ''.join(move)

ACTION_SPACE = open('features/chess_possible_moves.txt', 'r').read().splitlines()
MIRRORED_ACTIONS = [*map(mirror_uci, ACTION_SPACE)]
BLACK_TO_WHITE = {i: ACTION_SPACE.index(a) for i, a in enumerate(MIRRORED_ACTIONS)}
WHITE_TO_BLACK = {v: k for k, v in BLACK_TO_WHITE.items()}

UCI_TO_TENSOR = open('features/uci_to_tensor.txt', 'r').read().splitlines()
UCI_TO_TENSOR = [*map(int, UCI_TO_TENSOR)]
TENSOR_TO_UCI = {}
for i, v in enumerate(UCI_TO_TENSOR):
    if v in TENSOR_TO_UCI:
        TENSOR_TO_UCI[v].append(i)
    else:
        TENSOR_TO_UCI[v] = [i]

class Featurizer:
    def __init__(self):
        self.symbols = ['P', 'R', 'K', 'B', 'Q', 'K', 'p', 'r', 'k', 'b', 'q', 'k']

    def transform(self, state):
        matrices = []
        board = chess.Board(state)
        if not board.turn:
            board = board.mirror()

        for symbol in self.symbols:
            matrix = np.zeros((8, 8), dtype=int)
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.symbol() == symbol:
                    i, j = np.unravel_index(square, (8, 8))
                    matrix[i][j] = 1
            matrices.append(matrix)

        ones = np.ones((8, 8), dtype=int)
        zeros = np.zeros((8, 8), dtype=int)
        matrices.append(ones if board.has_kingside_castling_rights(chess.WHITE) else zeros)
        matrices.append(ones if board.has_queenside_castling_rights(chess.WHITE) else zeros)
        matrices.append(ones if board.has_kingside_castling_rights(chess.BLACK) else zeros)
        matrices.append(ones if board.has_queenside_castling_rights(chess.BLACK) else zeros)

        en_passant = np.zeros((8, 8), dtype=int)
        if board.ep_square is not None:
            i, j = np.unravel_index(board.ep_square, (8, 8))
            en_passant[i][j] = 1
        matrices.append(en_passant)

        halfmove_clock = np.full((8, 8), board.halfmove_clock, dtype=int)
        matrices.append(halfmove_clock)

        matrices = np.stack(matrices, axis=0)
        return matrices
    
    def create_tensor(self, policy):
        tensor = np.zeros(73*8*8, dtype=np.float32)
        for move, probability in enumerate(policy):
            tensor[UCI_TO_TENSOR[move]] = probability
        return tensor
    
    def unpack_tensor(self, tensor):
        policy = [0]*len(ACTION_SPACE)
        for i, probability in enumerate(tensor):
            if i in TENSOR_TO_UCI:
                for move in TENSOR_TO_UCI[i]:
                    policy[move] = probability
        return policy