import math
import random
import pickle
import os
import time
import chess

from neural_net import ACTION_SPACE, NeuralNetwork

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
        s = state.fen()
        if state.is_game_over():
            if state.is_checkmate():
                return -1
            else:
                return 0
        
        if s not in self.visited:
            self.Q_table[s] = {}
            legal_moves = [*map(str, state.legal_moves)]
            self.N[s] = {a: 1*(a in legal_moves) for a in ACTION_SPACE}
            value, p = model.evaluate(s)
            self.policy[s] = p
            self.visited.add(s)
            return -value
        
        a = max([*map(str, state.legal_moves)], key=lambda a: self.heuristic(s, a))

        state.push_uci(a)
        s2 = state
        value = self.search(s2, model)

        self.Q_table[s][a] = (self.N[s][a]*self.Q_table[s][a] + value)/(self.N[s][a]+1)
        self.N[s][a] += 1
        return -value
    
    def tree_policy(self, state, action_space):
        counts = self.N[state]
        total = sum(counts.values())
        return [counts[a]/total for a in action_space]


class Trainer:
    def __init__(self):
        self.model = NeuralNetwork()
        self.target = NeuralNetwork()

        self.threshold = 0.
        self.memory = []
        self.losses = []
        self.mcts = MCTS()
    
    def load(self, filename="target.pth"):
        if os.path.exists(filename):
            self.model.load_model(filename)
            self.target.load_model(filename)
        if os.path.exists('losses.pkl'):
            with open('losses.pkl', 'rb') as f:
                self.losses = pickle.load(f)
    
    def save(self):
        with open('losses.pkl', 'wb') as f:
            pickle.dump(self.losses, f)
        self.model.save_model("model.pth")

    def update_memory(self, episodes=10):
        for _ in range(episodes):
            self.memory.extend(self.run_episode(self.model))
        self.mcts = MCTS()
    
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

    def learn_policy(self, iterations, verbose=True):
        initial = time.perf_counter()
        for it in range(iterations):
            start = time.perf_counter()
            if verbose:
                print(f"\nsimulating trees {it+1}...")
            self.update_memory()
            if verbose:
                now = time.perf_counter()
                print(f"time taken per tree: {now-start:.1f}s\ntotal time: {now-initial:.1f}s\nmemory size: {len(self.memory)}")
            self.losses.append(self.model.update_weights(self.memory))
            if verbose:
                print(f"batch loss: {self.losses[-1]:.5f}")
            lose_rate = self.pit(verbose=True)
            
            if lose_rate <= self.threshold:
                self.target.load_state_dict(self.model.state_dict())
                self.target.save_model("target.pth")
            
            self.save()
            self.clear_memory()

    def run_episode(self, model, depth=100):
        state = chess.Board()
        snapshot = []
        
        while not state.is_game_over():
            s = state.fen()
            for _ in range(depth):
                self.mcts.search(chess.Board(s), model)
            policy = self.mcts.tree_policy(s, ACTION_SPACE)
            snapshot.append((s, policy, None))
            legal_moves = [*map(str, state.legal_moves)]
            weights = [p*(a in legal_moves) for p, a in zip(policy, ACTION_SPACE)]
            if sum(weights) == 0:
                a = random.choice(legal_moves)
            else:
                a = random.choices(ACTION_SPACE, weights=weights)[0]
            state.push_uci(a)

        reward = 0
        if state.is_checkmate():
            reward = 1
        
        snapshot = [(s, p, reward*[-1, 1][i%2==0]) for i, (s, p, _) in enumerate(snapshot[::-1])][::-1]
        return snapshot

if __name__ == "__main__":
    trainer = Trainer()
    trainer.load()
    trainer.learn_policy(iterations=1000, verbose=True)