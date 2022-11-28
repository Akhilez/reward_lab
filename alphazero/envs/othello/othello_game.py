import numpy as np
from alphazero.envs.othello.othello_logic import Board
from alphazero.lib.base_classes import Game


class OthelloGame(Game):
    square_content = {-1: "X", +0: "-", +1: "O"}

    @staticmethod
    def getSquarePiece(piece):
        return OthelloGame.square_content[piece]

    def __init__(self, n):
        self.n = n

    def get_init_board(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def get_board_size(self):
        # (a,b) tuple
        return (self.n, self.n)

    def get_action_size(self):
        # return number of actions
        return self.n * self.n + 1

    def get_next_state(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n * self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def get_valid_moves(self, board, player):
        # return a fixed size binary vector
        valids = [0] * self.get_action_size()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n * x + y] = 1
        return np.array(valids)

    def get_game_ended(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.countDiff(player) > 0:
            return 1
        return -1

    def get_canonical_form(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player * board

    def get_symmetries(self, board, pi):
        # mirror, rotational
        assert len(pi) == self.n**2 + 1  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def string_representation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(
            self.square_content[square] for row in board for square in row
        )
        return board_s

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")  # print the row #
            for x in range(n):
                piece = board[y][x]  # get the piece to print
                print(OthelloGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
