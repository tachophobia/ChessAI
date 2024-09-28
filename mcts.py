from model import ACTION_SPACE
import chess
import numpy as np


class MCTS:
    def __init__(self, exploration_factor=1, cache_size=10000):
        self.policy = {}
        self.Q_table = {}
        self.N = {}
        self.c = exploration_factor

        self.cache_size = int(cache_size)
        self.access_counter = 0
        self.state_access = {}
    
    # UCB1
    def heuristic(self, state, action):
        if action not in self.Q_table:
            self.Q_table[state][action] = 0.
        return self.Q_table[state][action] + \
               self.c * self.policy[state][ACTION_SPACE.index(action)] * \
               np.sqrt(sum(self.N[state].values()) / (1 + self.N[state][action]))
    
    def search(self, state, model):
        results = {}
        s0 = state.fen()
        stack = [(s0, None, None)]

        while stack:
            s, parent, prev_a = stack.pop()
            state = chess.Board(s)
            
            self.prune_cache()

            if s in results:
                if parent is not None:
                    value = results[s]
                    self.Q_table[parent][prev_a] = (self.N[parent][prev_a] * self.Q_table[parent][prev_a] + value) / (self.N[parent][prev_a] + 1)
                    self.N[parent][prev_a] += 1
                    results[parent] = -value
                continue
            
            stack.append((s, parent, prev_a))
            if state.is_game_over():
                results[s] = [0, -1][state.is_checkmate()]
                continue
            
            if s not in self.state_access:
                legal_moves = [*map(str, state.legal_moves)]
                self.Q_table[s] = {}
                self.N[s] = {a: 1*(a in legal_moves) for a in ACTION_SPACE}
                
                v, p = model.evaluate(s)                
                self.policy[s] = p

                if s == s0:
                    self.apply_dirichlet_noise(s)
                
                results[s] = -v

                self.state_access[s] = self.access_counter
                self.access_counter += 1
                continue
            
            a = max([*map(str, state.legal_moves)], key=lambda a: self.heuristic(s, a))
            state.push_uci(a)
            s2 = state.fen()
            stack.append((s2, s, a))

    def prune_cache(self):
        if len(self.state_access) >= self.cache_size:
            oldest_state = min(self.state_access, key=self.state_access.get)
            del self.Q_table[oldest_state]
            del self.N[oldest_state]
            del self.policy[oldest_state]
            del self.state_access[oldest_state]

    def select_action(self, state, tau=1):
        counts = self.N[state]
        visit_counts = np.array([counts[a] for a in ACTION_SPACE]) ** (1/tau)
        visit_counts /= visit_counts.sum()
        return np.random.choice(ACTION_SPACE, p=visit_counts)
    
    def action_probabilities(self, state):
        counts = self.N[state]
        total = sum(counts.values())
        return [counts[a] / total for a in ACTION_SPACE]
    
    def apply_dirichlet_noise(self, state: str, alpha=0.03):
        dirichlet_noise = np.random.dirichlet([alpha] * len(self.policy[state]))
        self.policy[state] = 0.75 * np.array(self.policy[state]) + 0.25 * dirichlet_noise