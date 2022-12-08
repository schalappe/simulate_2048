# 2048 game for Reinforcement Learning

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 2048 Game
The game of 2048 is a single player, perfect information, stochastic puzzle game. The board is
represented by a 4x4 grid with numbered tiles, and at each step the player has four possible actions
which correspond to the four arrow keys (up, down, right, left). hen the player selects an action,
all the tiles in the board slide in the corresponding direction until they reach the end of the board or
another tile of different value. Tiles of the same value are merged together to form a new tile with a
value equal to their sum, and the resulting value is added to the running score of the game. After each
move, a new tile randomly appears in an empty spot on the board with a value of 2 or 4. The game
ends when there are no more moves available to the player that can alter the board state.

## Aim
The goal was to train an agent to play the game 2048. This project was strongly inspired by
[AlphaZero](https://arxiv.org/pdf/1712.01815v1.pdf) and [Stochastic Muzero](https://openreview.net/pdf?id=X6D9bAHhBQ1).
At the creation of this project, two targets were set:

* Have a 2048 cell.
* to have a cell with a value greater than 2048

We reached the first targets after 300 self play games, not the second, even after 1500 self play games
played by the agent.

![2048 Game](game_short.gif)