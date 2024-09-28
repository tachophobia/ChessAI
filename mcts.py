from model import ACTION_SPACE
import chess
import numpy as np

class Node:
    def __init__(self, state: chess.Board, parent: tuple):
        self.parent, self.prev_action = parent

        self.state = state
        self.fen = state.fen()
        self.legal_moves = [*map(str, state.legal_moves)]

        self.children = []
        self.N = {} # action visits
        self.Q = {} # action values
        

class MCTS:
    def __init__(self, exploration_factor=1):
        self.policy = {}
        self.visited = {}
        self.c = exploration_factor
        self.root = None
    
    # UCB1
    def heuristic(self, node: Node, action: str):
        if action not in node.Q:
            node.Q[action] = 0.
        return node.Q[action] + self.c * self.policy[node.fen][ACTION_SPACE.index(action)] * \
                                np.sqrt(sum(node.N.values()) / (1 + node.N[action]))
    
    def search(self, model):
        results = {}
        stack = [self.root]

        while stack:
            node = stack.pop()

            if node.fen in results:
                if node.parent is not None:
                    value = results[node.fen]
                    prev_a = node.prev_action
                    node.parent.Q[prev_a] = (node.parent.N[prev_a] * node.parent.Q[prev_a] + value) / (node.parent.N[prev_a] + 1)
                    node.parent.N[prev_a] += 1
                    results[node.parent.fen] = -value
                continue
            
            stack.append(node)
            if node.state.is_game_over():
                results[node.fen] = [0, -1][node.state.is_checkmate()]
                continue
            
            if node.fen not in self.visited:
                v, p = model.evaluate(node.fen)
                node.N = {a: 1*(a in node.legal_moves) for a in ACTION_SPACE}
                self.policy[node.fen] = p
                if node == self.root:
                    self.apply_dirichlet_noise(node.fen)
                results[node.fen] = -v
                self.visited[node.fen] = node
                continue
            
            state = node.state.copy()
            node = self.visited[node.fen]
            
            a = max(node.legal_moves, key=lambda a: self.heuristic(node, a))
            state.push_uci(a)

            s2 = state.fen()
            child = next((c for c in node.children if c.fen == s2), None)
            if child is None:
                child = Node(state, (node, a))
                node.children.append(child)
            
            stack.append(child)

    def select_action(self, state, tau=1):
        counts = self.visited[state].N
        visit_counts = np.array([counts[a] for a in ACTION_SPACE]) ** (1/tau)
        visit_counts /= visit_counts.sum()
        return np.random.choice(ACTION_SPACE, p=visit_counts)
    
    def action_probabilities(self, state: str):
        counts = self.visited[state].N
        total = sum(counts.values())
        return [counts[a] / total for a in ACTION_SPACE]
    
    def apply_dirichlet_noise(self, state: str, alpha=0.03):
        dirichlet_noise = np.random.dirichlet([alpha] * len(self.policy[state]))
        self.policy[state] = 0.75 * np.array(self.policy[state]) + 0.25 * dirichlet_noise

    def destroy_tree(self, node):
        if node:
            if node.fen in self.visited: del self.visited[node.fen]
            if node.fen in self.policy: del self.policy[node.fen]
            for child in node.children:
                self.destroy_tree(child)
            del node

    def discard_above(self, state: str):
        if state in self.visited:
            new_root = self.visited[state]
            parent = new_root.parent
            start = len(self.visited)

            if parent:
                parent.children.remove(new_root)

            while parent:
                grandparent = parent.parent
                self.destroy_tree(parent)
                parent = grandparent

    
    def set_root(self, state):
        if state.fen() in self.visited:
            self.root = self.visited[state.fen()]
        else:
            self.root = Node(state, (None, None))