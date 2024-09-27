from collections import defaultdict
from model import ACTION_SPACE
import math
import chess

class MCTS:
    def __init__(self, exploration_factor=1, cache_size=20000):
        self.Q_table = defaultdict(lambda: defaultdict(float))
        self.N = defaultdict(lambda: defaultdict(int))
        self.policy = defaultdict(list)
        self.c = exploration_factor*len(ACTION_SPACE) # rescale by policy size
        
        self.cache_size = cache_size
        self.access_counter = 0
        self.state_access = {}
    
    # UCB1
    def heuristic(self, state, action):
        total_visits = sum(self.N[state].values())
        action_visits = self.N[state][action]
        exploration_term = self.c * self.policy[state][ACTION_SPACE.index(action)] * math.sqrt(total_visits / (1 + action_visits))
        return self.Q_table[state][action] + exploration_term
    
    def search(self, state, model):
        s0 = state.fen()
        results = {}
        stack = [(s0, None, None)]
        
        while stack:
            s, prev_s, prev_a = stack.pop()
            state = chess.Board(s)
            
            self.prune_cache()
            
            if s in results:
                if prev_s is not None:
                    value = results[s]
                    self.update_q_table(prev_s, prev_a, value)
                    results[prev_s] = -value
                continue
            
            stack.append((s, prev_s, prev_a))
            if state.is_game_over():
                results[s] = [0, -1][state.is_checkmate()]
                continue
            
            if s not in self.Q_table:
                legal_moves = [move.uci() for move in state.legal_moves]
                v, p = model.evaluate(s)

                self.N[s] = {a: 1*(a in legal_moves) for a in ACTION_SPACE}
                self.policy[s] = p
                results[s] = -v

                self.state_access[s] = self.access_counter
                self.access_counter += 1
                continue
            
            a = max([move.uci() for move in state.legal_moves], key=lambda a: self.heuristic(s, a))
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
    
    def update_q_table(self, prev_s, prev_a, value):
        self.Q_table[prev_s][prev_a] = (self.N[prev_s][prev_a] * self.Q_table[prev_s][prev_a] + value) / (self.N[prev_s][prev_a] + 1)
        self.N[prev_s][prev_a] += 1
    
    def tree_policy(self, state, action_space):
        counts = self.N[state]
        total = sum(counts.values())
        return [counts[a] / total for a in action_space]