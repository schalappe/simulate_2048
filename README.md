# 2048 game for Reinforcement Learning

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 2048 Game
The game of 2048 is a single player, perfect information, stochastic puzzle game. The board is represented by a 4x4
grid of numbered tiles, and at each step the player has four possible actions, which correspond to the four arrow keys
(up, down, right, left). When the player chooses an action, all the tiles on the board slide in the corresponding
direction until they reach the end of the board or another tile of a different value. Tiles of the same value are
combined to form a new tile with a value equal to their sum, and the resulting value is added to the running score of
the game. After each move, a new tile with a value of 2 or 4 appears randomly in an empty space on the board. The game
ends when the player has no more moves that can change the state of the board.

## Aim
The goal was to train an agent to play the game 2048. This project was heavily inspired by
[AlphaZero](https://arxiv.org/pdf/1712.01815v1.pdf) and [Stochastic Muzero](https://openreview.net/pdf?id=X6D9bAHhBQ1).
When this project was created, two goals were set:

* To have a 2048 cell.
* Have a cell with a value greater than 2048

We achieved the first goal after 300 self-plays, but not the second, even after 1500 self-plays of the agent.

![2048 Game](docs/imgs/game_short.gif)