# QLearning-TicTacToe
Using QLearning to play a game of Tic-Tac-Toe against 3 types of opponents: Random, Smart, and Perfect Players

#### 6 Files:
- Board.py: code to make tic-tac-toe board, 3 by 3 grid
- RandomPlayer: moves randomly
- SmartPlayer: better than RandomPlayer, but cannot beat PerfectPlayer
- PerfectPlayer: never loses
- QLearner.py: Q-Learning Player Code
- TicTacToe.py: driver code, where all players play tic-tac-toe games and where our QLearner is trained and tested

#### Q-Learning:
Q ( s , a ) ← (1 - ⍺) Q ( s , a ) + ⍺ ( R ( s ) + 𝛾 max a ’ Q ( s’ , a ’))

#### Parameters:
Reward for WIN,DRAW,LOSE, learning rate, discount factor and initial conditions.
