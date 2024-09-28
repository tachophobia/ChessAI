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
        self.target = Model()
        self.target.load_state_dict(self.model.state_dict())
        self.cpu_count = psutil.cpu_count()

        self.threshold = 0.
        self.experience = []
        self.losses = []
        self.verbose = verbose
        self.path = training_directory
    
    def load(self):
        self.model.load_model(self.path+'model.pth')
        if os.path.exists(self.path+'target.pth'):
            self.target.load_model(self.path+'target.pth')
        if os.path.exists(self.path+'losses.pkl'):
            with open(self.path+'losses.pkl', 'rb') as f:
                self.losses = pickle.load(f)
    
    def save(self):
        with open(self.path+'losses.pkl', 'wb') as f:
            pickle.dump(self.losses, f)
        with open(self.path+'experience.pkl', 'wb') as f:
                pickle.dump(self.experience, f)
        self.model.save_model(self.path+"model.pth")

    def pit(self, games=50):
        wins = 0
        draws = 0
        moves = []
        with multiprocessing.Pool(processes=self.cpu_count) as pool:
            results = pool.starmap(self.run_game, [(10, False) for _ in range(games)])

        for winner, move in results:
            if winner == 1:
                wins += 1
            elif winner == 0:
                draws += 1
            moves.append(move)
        lose_rate = (games-wins-draws)/games

        if self.verbose:
            win_rate, draw_rate, avg_moves = wins/games, draws/games, sum(moves)/games
            print(f"simulated {games} games")
            print(f"win rate: {win_rate:.2f}, draw rate: {draw_rate:.2f}, lose rate: {lose_rate:.2f}, avg moves: {avg_moves:.2f}")
        return lose_rate
    
    def run_game(self, depth, collect_state=False):
        state = chess.Board()
        player = True
        white_model, black_model = self.model, self.target

        if random.random() < 0.5:
            white_model, black_model = black_model, white_model
            player = not player

        white_mcts = MCTS()
        black_mcts = MCTS()

        move_count = 0

        while not state.is_game_over():
            if state.turn:
                model = white_model
                mcts = white_mcts
            else:
                model = black_model
                mcts = black_mcts
                move_count += 1

            mcts.prune_tree(state.fen())
            for _ in range(depth):
                mcts.search(state.copy(), model)
            
            move = mcts.select_action(state.fen(), tau=1e-2)
            state.push_uci(move)

        if state.is_checkmate():
            outcome = [-1, 1][state.outcome().winner == player]
        else:
            outcome = 0

        if collect_state:
            return state, player, outcome
        return outcome, move_count

    
    def learn_policy(self, iterations, trees=200):
        iterator = range(iterations)
        trees = trees//self.cpu_count
        if self.verbose:
            print("\nSTARTING POLICY ITERATION")
            print(f"trees per episode: {self.cpu_count*trees}\nconcurrent processes: {self.cpu_count}\n")
            iterator = tqdm(iterator)
        
        for it in iterator:
            process = range(trees)
            if self.verbose: 
                print(f"\n\nsimulating trees ({it})")
                process = tqdm(process)
            results = []

            with multiprocessing.Pool(processes=self.cpu_count) as pool:
                for _ in process:
                    results.extend(pool.starmap(self.run_episode, [(self.target,) for _ in range(self.cpu_count)]))
                gc.collect()

            for game in results:
                self.experience.extend(game)
            random.shuffle(self.experience)

            if self.verbose:
                print(f"\nexperience size: {len(self.experience)}")
            
            batch_losses = self.model.update_weights(self.experience, verbose=self.verbose)
            self.losses.extend(batch_losses)
            if self.verbose:
                print(f"\nbatch loss: {sum(batch_losses)/len(batch_losses):.5f}")

            lose_rate = self.pit()
            if lose_rate <= self.threshold:
                self.target.load_state_dict(self.model.state_dict())
                self.target.save_model(self.path+"target.pth")
                self.model.reset_optimizer()
            
            self.save()
            self.experience = []

    def run_episode(self, model, depth=80):
        state = chess.Board()
        snapshot = []

        tree = MCTS()
        while not state.is_game_over():
            s = state.fen()
            tree.prune_tree(s)

            for _ in range(depth):
                tree.search(chess.Board(s), model)

            policy = tree.action_probabilities(s)
            snapshot.append((s, policy))

            a = tree.select_action(s, [1e-2, 1][len(snapshot) <= 30])
            state.push_uci(a)

        reward = 0
        if state.is_checkmate():
            reward = 1
        
        snapshot = [(s, p, reward*[-1, 1][i%2==0]) for i, (s, p) in enumerate(snapshot[::-1])][::-1]
        return snapshot

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    trainer = Trainer(verbose=True, training_directory="belisarius/")
    # trainer.load()
    trainer.learn_policy(iterations=80)