import math
import random
import pickle
import os
import chess
import time
from neural_net import ACTION_SPACE, NeuralNetwork
from tqdm import tqdm

import multiprocessing
import psutil
import gc

class MCTS:
    def __init__(self, exploration_factor=1):
        self.Q_table = {}
        self.N = {}
        self.policy = {}
        self.visited = set()
        self.c = exploration_factor
    
    # UCB1
    def heuristic(self, state, action):
        if action not in self.Q_table[state]:
            self.Q_table[state][action] = 0.
        return self.Q_table[state][action] + \
               self.c * self.policy[state][ACTION_SPACE.index(action)] * \
               math.sqrt(sum(self.N[state].values())/(1+self.N[state][action]))
    
    def search(self, state, model):
        s0 = state.fen()
        stack = [(s0, None, None)]
        results = {}
        
        while stack:
            s, prev_s, prev_a = stack.pop()
            state = chess.Board(s)

            if s in results:
                if prev_s is not None:
                    value = results[s]
                    self.Q_table[prev_s][prev_a] = (self.N[prev_s][prev_a]*self.Q_table[prev_s][prev_a] + value)/(self.N[prev_s][prev_a]+1)
                    self.N[prev_s][prev_a] += 1
                    results[prev_s] = -value
                continue

            stack.append((s, prev_s, prev_a))
            if state.is_game_over():
                results[s] = [0, -1][state.is_checkmate()]
                continue
            
            if s not in self.visited:
                self.Q_table[s] = {}
                legal_moves = [*map(str, state.legal_moves)]
                self.N[s] = {a: 1*(a in legal_moves) for a in ACTION_SPACE}
                value, p = model.evaluate(s)
                self.policy[s] = p
                self.visited.add(s)
                results[s] = -value
                continue
            
            a = max([*map(str, state.legal_moves)], key=lambda a: self.heuristic(s, a))

            state.push_uci(a)
            s2 = state.fen()
            stack.append((s2, s, a))

        return results[s0]
    
    def tree_policy(self, state, action_space):
        counts = self.N[state]
        total = sum(counts.values())
        return [counts[a]/total for a in action_space]


class Trainer:
    def __init__(self):
        self.model = NeuralNetwork()
        self.target = NeuralNetwork()
        self.target.load_state_dict(self.model.state_dict())

        self.threshold = 0.
        self.memory = []
        self.losses = []
    
    def load(self):
        self.model.load_model("model.pth")
        if os.path.exists("target.pth"):
            self.target.load_model("target.pth")
        if os.path.exists('losses.pkl'):
            with open('losses.pkl', 'rb') as f:
                self.losses = pickle.load(f)
    
    def save(self):
        with open('losses.pkl', 'wb') as f:
            pickle.dump(self.losses, f)
        self.model.save_model("model.pth")
    
    def clear_memory(self):
        self.memory = []

    def pit(self, games=50, verbose=False):
        wins = 0
        draws = 0
        moves = []

        results = [self.run_game() for _ in range(games)]

        for winner, move in results:
            if winner == 1:
                wins += 1
            elif winner == 0:
                draws += 1
            moves.append(move)
        losses = games-wins-draws
        lose_rate = losses/games

        if verbose:
            win_rate, draw_rate, avg_moves = wins/games, draws/games, sum(moves)/games
            print(f"win rate: {win_rate:.2f}, draw rate: {draw_rate:.2f}, lose rate: {lose_rate:.2f}, avg moves: {avg_moves:.2f}")
        return lose_rate
    
    def run_game(self, collect_state=False):
        state = chess.Board()

        player = True
        white, black = self.model, self.target
        if random.random() < 0.5:
            white, black = black, white
            player = not player
        move = 0
        while not state.is_game_over():
            if state.turn:
                a = white.choose_move(state)
            else:
                a = black.choose_move(state)
                move += 1
            state.push_uci(a)
        
        if state.is_checkmate():
            outcome = [-1, 1][state.outcome().winner == player]
        else:
            outcome = 0
        
        if collect_state:
            return state, player, outcome
        return outcome, move

    def worker(self, params, cpu):
        psutil.Process().cpu_affinity([cpu])
        return self.run_episode(params)
    
    def learn_policy(self, iterations, trees=100, verbose=True):
        iterator = range(iterations)
        if verbose:
            iterator = tqdm(iterator)

        cpu_count = psutil.cpu_count(logical=False)
        processes = cpu_count - (trees % cpu_count)
        with multiprocessing.Pool(processes=processes) as pool:
            for it in iterator:
                if verbose: 
                    print(f"\nsimulating trees {it}...")

                start = time.perf_counter()
                
                for _ in range(0, trees, processes):
                    results = pool.starmap(
                        self.worker, [(self.model, i) for i in range(processes)])[0]
                    self.memory.extend(results)
                random.shuffle(self.memory)

                if verbose:
                    delta = time.perf_counter()-start
                    print(f"time taken per tree: {delta/trees:.1f}s\ntotal time: {delta:.1f}s\nmemory size: {len(self.memory)}")
                
                batch_losses = self.model.update_weights(self.memory)
                self.losses.extend(batch_losses)

                if verbose:
                    print(f"batch loss: {sum(batch_losses)/len(batch_losses):.5f}")
                lose_rate = self.pit(verbose=True)
                
                if lose_rate <= self.threshold:
                    self.target.load_state_dict(self.model.state_dict())
                    self.target.save_model("target.pth")
                
                self.save()
                self.clear_memory()
                gc.collect()
        
        pool.close()
        pool.join()

    def run_episode(self, model, depth=25):
        state = chess.Board()
        snapshot = []
        mcts = MCTS()
        while not state.is_game_over():
            s = state.fen()
            for _ in range(depth):
                mcts.search(chess.Board(s), model)
            policy = mcts.tree_policy(s, ACTION_SPACE)
            snapshot.append((s, policy, None))
            a = random.choices(ACTION_SPACE, weights=policy)[0]
            state.push_uci(a)

        reward = 0
        if state.is_checkmate():
            reward = 1
        
        snapshot = [(s, p, reward*[-1, 1][i%2==0]) for i, (s, p, _) in enumerate(snapshot[::-1])][::-1]
        return snapshot

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    trainer = Trainer()
    trainer.learn_policy(iterations=80, verbose=True)