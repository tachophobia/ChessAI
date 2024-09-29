import chess
import numpy as np
import random
from model import ACTION_SPACE


class MCTS:
    def __init__(self, exploration_factor=1):
        self.policy = {}
        self.Q_table = {}
        self.N = {}
        self.c = exploration_factor
        
        self.parent_map = {}
        self.visited = set()

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
            
            if parent is not None:
                if parent not in self.parent_map:
                    self.parent_map[parent] = []
                if s not in self.parent_map[parent]:
                    self.parent_map[parent].append(s)

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
            
            if s not in self.visited:
                legal_moves = [*map(str, state.legal_moves)]
                self.Q_table[s] = {}
                self.N[s] = {a: 1*(a in legal_moves) for a in ACTION_SPACE}
                
                v, p = model.evaluate(s)                
                self.policy[s] = p

                if s == s0:
                    self.apply_dirichlet_noise(s)
                
                results[s] = -v

                self.visited.add(s)
                continue
            
            a = max([*map(str, state.legal_moves)], key=lambda a: self.heuristic(s, a))
            state.push_uci(a)
            s2 = state.fen()
            stack.append((s2, s, a))

    def select_action(self, state, greedy=False):
        counts = [self.N[state][a] for a in ACTION_SPACE]
        if not greedy:
            return random.choices(ACTION_SPACE, weights=counts)[0]
        else:
            return ACTION_SPACE[np.argmax(counts)]
    
    def action_probabilities(self, state):
        counts = self.N[state]
        total = sum(counts.values())
        return [counts[a] / total for a in ACTION_SPACE]
    
    def apply_dirichlet_noise(self, state: str, alpha=0.03, epsilon=0.25):
        dirichlet_noise = np.random.dirichlet([alpha] * len(self.policy[state]))
        self.policy[state] = (1-epsilon) * np.array(self.policy[state]) + epsilon * dirichlet_noise

    def collect_subtree(self, state):
        subtree = {state}
        if state in self.parent_map:
            for child in self.parent_map[state]:
                subtree.update(self.collect_subtree(child))
        return subtree

    def prune_tree(self, new_root):
        if new_root in self.visited:
            subtree = self.collect_subtree(new_root)
            states_to_remove = [state for state in self.visited if state not in subtree]
            for state in states_to_remove:
                self.remove_state(state)

    def remove_state(self, state):
        self.Q_table.pop(state, None)
        self.N.pop(state, None)
        self.policy.pop(state, None)

        if state in self.parent_map:
            del self.parent_map[state]
        
        for _, children in self.parent_map.items():
            if state in children:
                children.remove(state)

        self.visited.discard(state)

