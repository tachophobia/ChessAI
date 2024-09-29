import chess
import random
import pickle
import os
import torch

import multiprocessing
import psutil
import gc

from model import Model
from mcts import MCTS
from tqdm import tqdm


class Trainer:
    def __init__(self, verbose=False, training_directory=""):
        self.model = Model()
        self.cpu_count = psutil.cpu_count()

        self.threshold = 0.
        self.experience = []
        self.losses = []
        self.verbose = verbose
        self.path = training_directory
    
    def load(self):
        if os.path.exists(self.path+'model.pkl'):
            self.model.load_model(self.path+"model.pth")
        if os.path.exists(self.path+'losses.pkl'):
            with open(self.path+'losses.pkl', 'rb') as f:
                self.losses = pickle.load(f)
    
    def save(self):
        with open(self.path+'losses.pkl', 'wb') as f:
            pickle.dump(self.losses, f)
        with open(self.path+'experience.pkl', 'wb') as f:
                pickle.dump(self.experience, f)
        self.model.save_model(self.path+"model.pth")

    def pit(self, opponent, games=50):
        wins = 0
        draws = 0
        moves = []

        results = []
        games = games//self.cpu_count * self.cpu_count
        iterator = range(games//self.cpu_count)
        if self.verbose:
            print(f"\nPitting agents in {games} games\n")
            iterator = tqdm(iterator)
        
        with multiprocessing.Pool(processes=self.cpu_count) as pool:
            for _ in iterator:
                results.extend(pool.starmap(self.run_game, [(opponent, 10, False) for _ in range(self.cpu_count)]))
                gc.collect()

        for winner, move in results:
            if winner == 1:
                wins += 1
            elif winner == 0:
                draws += 1
            moves.append(move)
        lose_rate = (games-wins-draws)/games

        if self.verbose:
            win_rate, draw_rate, avg_moves = wins/games, draws/games, sum(moves)/games
            print(f"win rate: {win_rate:.2f}, draw rate: {draw_rate:.2f}, lose rate: {lose_rate:.2f}, avg moves: {avg_moves:.2f}")
        return lose_rate
    
    def run_game(self, opponent, depth, collect_state=False):
        state = chess.Board()
        player = True
        white_model, black_model = self.model, opponent

        if random.random() < 0.5:
            white_model, black_model = black_model, white_model
            player = not player

        white_mcts = MCTS(exploration_factor=0)
        black_mcts = MCTS(exploration_factor=0)

        move_count = 0

        while not state.is_game_over():
            if state.turn:
                model = white_model
                mcts = white_mcts
            else:
                model = black_model
                mcts = black_mcts
                move_count += 1

            if model == 'random':
                state.push_uci(random.choice([*map(str, state.legal_moves)]))
                continue

            mcts.prune_tree(state.fen())
            for _ in range(depth):
                mcts.search(state.copy(), model)
            
            move = mcts.select_action(state.fen(), greedy=False)
            state.push_uci(move)

        if state.is_checkmate():
            outcome = [-1, 1][state.outcome().winner == player]
        else:
            outcome = 0

        if collect_state:
            return state, player, outcome
        return outcome, move_count

    
    def learn_policy(self, iterations, games_per_iter=300):
        iterator = range(iterations)
        games = games_per_iter//self.cpu_count
        if self.verbose:
            print("\nSTARTING POLICY ITERATION")
            print(f"games per iteration: {self.cpu_count*games}\nconcurrent processes: {self.cpu_count}\n")
            iterator = tqdm(iterator)
        
        for it in iterator:
            process = range(games)
            if self.verbose: 
                print(f"\n\nsimulating trees ({it})")
                process = tqdm(process)
            results = []

            for _ in process:
                with multiprocessing.Pool(processes=self.cpu_count) as pool:
                    results.extend(pool.starmap(self.run_episode, [(self.model,) for _ in range(self.cpu_count)]))
            
            for game in results: self.experience.extend(game)
            random.shuffle(self.experience)
            self.losses.extend(self.model.update_weights(self.experience, verbose=self.verbose))

            if self.verbose:
                print(f"\nexperience size: {len(self.experience)}")
                print(f"\nbatch loss: {self.losses[-1]:.5f}")
            
            self.save()
            self.experience = []
            gc.collect()
                   

    def run_episode(self, model, depth=50, max_len=512):
        state = chess.Board()
        snapshot = []
        tree = MCTS()
        while not state.is_game_over() and len(snapshot) < max_len:
            s = state.fen()
            tree.prune_tree(s)

            for _ in range(depth):
                tree.search(chess.Board(s), model)

            policy = tree.action_probabilities(s)
            snapshot.append((s, policy))

            a = tree.select_action(s, [True, False][len(snapshot) <= 30])
            state.push_uci(a)

        reward = 0
        if state.is_checkmate():
            reward = 1
        
        snapshot = [(s, p, reward*[-1, 1][i%2==0]) for i, (s, p) in enumerate(snapshot[::-1])][::-1]
        return random.choices(snapshot, k=min(30, len(snapshot)))

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    trainer = Trainer(verbose=True, training_directory="belisarius/")
    trainer.load()
    # trainer.pit('random')
    trainer.learn_policy(iterations=80)