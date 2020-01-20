from Board import Board
import numpy as np
import collections
import random

# -------------------------------------------------------


class QLearner:

    # ======================================================================
    GAME_NUM = 100000  # number of games you want to train qlearner
    # ======================================================================

    def __init__(self):

        self.alpha = 0.6  # learning rate
        self.gamma = 0.8  # discount rate

        self.win_reward = 1.0
        self.draw_reward = 0.5
        self.loss_reward = -1.0

        self.epsilon = 0.1  # probability of choosing between exploration and exploitation

        # Decay/reduce epsilon gradually, as initially we want to output random moves, so the model sees as many states
        # as possible, because when epoch run is low, q_value for most states would be 0.6, but as the epochs increase
        # q_table gets filled and thus we need to output random moves less frequently, and encourage exploitation
        self.eps_decay = 0.001

        # all possible moves/cell_indexes in a 3x3 TicTacToe
        self.all_moves = []
        for i in range(3):
            for j in range(3):
                self.all_moves.append((i, j))

        # Initialize Q Table to hold a dictionary corresponding to all actions possible in a state
        self.QTable = {}

        for move in self.all_moves:

            # Default dict so if item not in dict, it will be added
            # Initialize with 0.6 to encourage exploration
            self.QTable[move] = collections.defaultdict(lambda: 0.6)

        # QTable = {<move>: {<board state>: <val(init=0.6)>}}
        # stores each move as key to a dict with all board states move sees in its lifetime
        # Total moves: 9; Total states: 3**9 as ('O', 'X', '-'), and total 9 cells

        # List of all states seen in a game, used to update QTable in learn()
        self.states_seen = []

        # List of all moves taken in a game, used to update QTable in learn()
        self.moves_made = []

    # =============================================================================

    def move(self, board):
        """
        given the board, make the 'best' move
        see `play()` method in TicTacToe.py
        Parameters: board
        """

        if board.game_over():
            return

        # Encode stat of board, eg: '122100211'
        curr_state = board.encode_state()

        # -------------------------------------------------------
        # list of all possible moves
        moves_available = []

        # list of all QTable values
        q_values = np.array([])

        for mov in self.all_moves:

            if board.is_valid_move(mov[0], mov[1]):
                q_values = np.append(q_values, self.QTable[mov][curr_state])
                moves_available.append(mov)

            else:
                # Assign all non-valid moves a -inf value
                self.QTable[mov][curr_state] = float('-inf')

        # -------------------------------------------------------
        # Select move

        # Exploration
        if random.random() < self.epsilon:
            chosen_idx = np.random.randint(len(moves_available))

        # Exploitation
        else:
            # Indexes of max values
            max_index = np.where(q_values == np.max(q_values))[0]

            if len(max_index) > 1:
                # if more than one max values is same
                chosen_idx = np.random.choice(max_index, 1)[0]

            else:
                # if only one max value, take that
                chosen_idx = max_index[0]

        move = moves_available[chosen_idx]
        # -------------------------------------------------------

        self.states_seen.append(curr_state)
        self.moves_made.append(move)

        self.epsilon *= (1.0 - self.eps_decay)

        # -------------------------------------------------------
        return board.move(move[0], move[1], self.side)

    # =============================================================================

    def learn(self, board):
        """
        when game ends, this method will be called to learn from the previous game i.e. update QValues
        see `play()` method in TicTacToe.py
        Parameters: board
        """

        if board.game_result == 0:
            reward = self.draw_reward
        elif board.game_result == self.side:
            reward = self.win_reward
        else:
            reward = self.loss_reward

        # Reverse lists, to help propagate up reward values
        self.states_seen = self.states_seen[::-1]
        self.moves_made = self.moves_made[::-1]

        # -------------------------------------------------------
        # QLearning Equation:
        # Q(s,a) = Q(s,a) + alpha * [r + gamma * maxQ(s',a') - Q(s,a)]
        # Q(s,a) = Current values of QTable
        # alpha = Learning Rate
        # gamma = Discount Rate
        # r = Reward earned from move
        # maxQ(s',a') = Total future reward based on actions in the next state
        # -------------------------------------------------------
        next_state_q = 0  # Total future reward values for next state

        for idx in range(len(self.moves_made)):

            if idx == 0:

                # final state: Q(s,a) = Q(s,a) + alpha[reward - Q(s,a)], as no next state, and thus no discount as well
                self.QTable[self.moves_made[idx]][self.states_seen[idx]] += \
                    self.alpha * (reward - self.QTable[self.moves_made[idx]][self.states_seen[idx]])

            else:
                self.QTable[self.moves_made[idx]][self.states_seen[idx]] += self.alpha * \
                            (self.gamma * max(next_state_q) - self.QTable[self.moves_made[idx]][self.states_seen[idx]])

            next_state_q = [self.QTable[move][self.states_seen[idx]] for move in self.all_moves]

        # -------------------------------------------------------

        # Re-init to empty
        self.states_seen = []
        self.moves_made = []

    # =============================================================================

    # do not change this function
    def set_side(self, side):
        # side = 2 or 1, based on player is 'O' or 'X' respectively
        self.side = side