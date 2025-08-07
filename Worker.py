from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet import NNetWrapper as nn
from othello.pytorch.NNet_cpu import NNetWrapper as nn_cpu
from utils import *
import ray
import numpy as np
from MCTS import MCTS

@ray.remote(num_gpus=0.25)
class Worker:
    def __init__(self):
        self.game = Game(6)
        self.nnet = nn(self.game)
        self.args = dotdict({
            'numIters': 1000,
            'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
            'tempThreshold': 15,        #
            'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
            'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
            'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
            'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
            'cpuct': 1,

            'checkpoint': './temp/',
            'load_model': False,
            'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
            'numItersForTrainExamplesHistory': 20,
        })

    def run_episodes(self, num_eps):
        result = []
        for _ in range(num_eps):
            mcts = MCTS(self.game, self.nnet, self.args)

            trainExamples = []
            board = self.game.getInitBoard()
            curPlayer = 1
            episodeStep = 0

            while True:
                episodeStep += 1
                canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
                temp = int(episodeStep < self.args.tempThreshold)

                pi = mcts.getActionProb(canonicalBoard, temp=temp)
                sym = self.game.getSymmetries(canonicalBoard, pi)
                for b, p in sym:
                    trainExamples.append([b, curPlayer, p, None])

                action = np.random.choice(len(pi), p=pi)
                board, curPlayer = self.game.getNextState(board, curPlayer, action)

                r = self.game.getGameEnded(board, curPlayer)

                if r != 0:
                    result += [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]
                    break
        return result