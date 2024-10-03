import chess
import chess.engine
import numpy as np
from features import ACTION_SPACE

class Engine:
    def __init__(self, path="stockfish/stockfish-ubuntu-x86-64-avx2", cp_scale=3e2):
        self.path = path
        self.scale = cp_scale
        self.depth = 1   

    def start(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
        self.engine.configure({"Threads": 2})

    def stop(self):
        self.engine.quit()

    def evaluate(self, fen, normalize=True):
        state = chess.Board(fen)

        info = self.engine.analyse(state, chess.engine.Limit(depth=self.depth))
        score = info['score'].relative
        if score.is_mate():
            value = np.sign(score.mate()) * 1e5
        else:
            value = np.tanh(score.score() / self.scale)

        p = [0.] * len(ACTION_SPACE)
        legal_moves = [move.uci() for move in state.legal_moves]

        for a in legal_moves:
            state.push_uci(a)
            info = self.engine.analyse(state, chess.engine.Limit(depth=self.depth))
            score = info['score'].relative
            v = score.score()
            if v is None:
                v = score.mate() * 1e5
            v *= -1
            p[ACTION_SPACE.index(a)] = 1 / (1 + np.exp(-v / self.scale, dtype=np.float128))
            state.pop()

        if normalize:
            p = np.array(p)
            p /= p.sum()
            p = p.tolist()

        # for i, prob in enumerate(p):
        #     if prob != 0:
        #         print(f"{ACTION_SPACE[i]}: {prob}")
        return value, p


if __name__ == "__main__":
    engine = Engine()
    engine.evaluate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
