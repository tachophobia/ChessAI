import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast

import numpy as np
import chess
import random

rank_map = {
    '1': '8', '2': '7', '3': '6', '4': '5',
    '5': '4', '6': '3', '7': '2', '8': '1'
}

def mirror_uci(move):
        move = [*move]
        move[1], move[3] = rank_map[move[1]], rank_map[move[3]]
        return ''.join(move)

ACTION_SPACE = open('chess_possible_moves.txt', 'r').read().splitlines()
MIRRORED_ACTIONS = [*map(mirror_uci, ACTION_SPACE)]
TRANSITIONS = [ACTION_SPACE.index(a) for a in MIRRORED_ACTIONS]

class Featurizer:
    def __init__(self):
        self.symbols = ['P', 'R', 'K', 'B', 'Q', 'K', 'p', 'r', 'k', 'b', 'q', 'k']

    def featurize(self, state):
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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.batchnorm1(self.conv1(x)))
        out = self.batchnorm2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class NeuralNetwork(nn.Module):
    def __init__(self, lr=1e-3, residual_blocks=20, batch_size=64):
        super(NeuralNetwork, self).__init__()
        
        self.initial_conv = nn.Conv2d(18, 64, kernel_size=3, padding=1)
        self.initial_batchnorm = nn.BatchNorm2d(64)
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(residual_blocks)]
        )
        
        self.fc_v1 = nn.Linear(64 * 8 * 8, 256)
        self.fc_v2 = nn.Linear(256, 1) 
        self.fc_p1 = nn.Linear(64 * 8 * 8, 256) 
        self.fc_p2 = nn.Linear(256, 1968)

        self.featurizer = Featurizer()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        torch.backends.cuda.matmul.allow_tf32 = True
        self.scaler = GradScaler()

    def forward(self, x):
        x = self.featurizer.featurize(x)
        x = x[np.newaxis, :, :, :]
        x = torch.from_numpy(x).float().to(self.device)
        x = F.relu(self.initial_batchnorm(self.initial_conv(x)))
        
        x = self.residual_blocks(x)
        
        x = x.view(x.size(0), -1)
        
        v = F.relu(self.fc_v1(x))
        v = torch.tanh(self.fc_v2(v))
        
        p = F.relu(self.fc_p1(x))
        p = F.softmax(self.fc_p2(p), dim=1)
        
        return v, p
    
    def evaluate(self, x):
        self.eval()
        with torch.no_grad():
            v, p = self(x)
            p = p.tolist()[0]
            if x.split()[1] == 'b':
                p = sorted(p, key=lambda a: TRANSITIONS[p.index(a)])
            return v, p
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
    
    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))

    def update_weights(self, targets):
        self.train()
        losses = []
        
        for i in range(0, len(targets), self.batch_size):
            batch = targets[i:i + self.batch_size]
            
            total_loss = 0.
            
            for state, policy, reward in batch:
                if state.split()[1] == 'b':
                    # policy action probs are in white basis, convert to black basis
                    policy = sorted(policy, key=lambda a: TRANSITIONS.index(policy.index(a)))
                with autocast():
                    predicted_v, predicted_p = self(state)
                    policy = torch.tensor(policy).to(self.device)
                    reward = torch.tensor([[reward]]).to(self.device)
                    
                    value_loss = torch.square(predicted_v - reward).mean()
                    policy_loss = -torch.sum(policy * torch.log(predicted_p + 1e-8), dim=1).mean()

                    total_loss += value_loss
                    total_loss += policy_loss
            
            total_loss /= len(batch)
            losses.append(float(total_loss))
            
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            del total_loss, value_loss, policy_loss, predicted_v, predicted_p
            torch.cuda.empty_cache()
        
        return np.mean(losses)
    
    def choose_move(self, state):
        mirrored = False
        if not state.turn:
            state = state.mirror()
            mirrored = True
        self.eval()
        with torch.no_grad():
            _, p = self(state.fen())
            p = p.tolist()[0]
            legal_moves = {*map(str, state.legal_moves)}
            action_space = [ACTION_SPACE, MIRRORED_ACTIONS][mirrored]
            p = [(p, a) for p, a in zip(p, action_space) if a in legal_moves]
            weights = [p[0] for p in p]
            actions = [p[1] for p in p]
            if sum(weights) == 0:
                weights = [1] * len(weights)
            action = random.choices(actions, weights=weights)[0]
            if mirrored:
                return mirror_uci(action)
            return action
        
