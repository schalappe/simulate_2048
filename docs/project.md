# Project Documentation: 2048 AI Agent

This document provides a comprehensive overview of the 2048 AI agent project, which implements the strategies outlined in the paper "Planning in Stochastic Environments with a Learned Model." The project uses a Monte Carlo Tree Search (MCTS) algorithm guided by a neural network to play the game of 2048, with the game environment implemented using Gymnasium.

## Project Goal

The primary goal of this project is to replicate the findings of the research paper and develop an AI agent that can effectively play the game of 2048. The agent uses a learned model of the game's dynamics to plan its moves, leveraging the power of MCTS to explore the game tree and make optimal decisions.

## Codebase Structure

The project is organized into several key directories, each with a specific purpose:

- **`twentyfortyeight/`**: The core of the project, containing the 2048 game logic and environment.
- **`monte_carlo/`**: Implements the Monte Carlo Tree Search algorithm.
- **`neurals/`**: Contains the neural network models used by the agent.
- **`notebooks/`**: Includes Jupyter notebooks for experimentation and analysis.
- **`tests/`**: Unit tests for the various components of the project.

### `twentyfortyeight/`

This directory contains the 2048 game implementation, which is divided into three subdirectories:

- **`core/`**: Implements the fundamental game logic, including:
    - `gameboard.py`: Defines the game board, its state, and the rules of the game.
    - `gamemove.py`: Manages the application of moves to the game board.
- **`envs/`**: Contains the Gymnasium environment for the 2048 game.
    - `twentyfortyeight.py`: Implements the `gym.Env` interface, allowing the agent to interact with the game in a standardized way.
- **`utils/`**: Provides helper functions and utilities used throughout the project.

### `monte_carlo/`

This directory contains the implementation of the MCTS algorithm, which is central to the agent's decision-making process. The key files are:

- **`actor.py`**: Defines the actor, which is responsible for selecting moves based on the MCTS search results.
- **`node.py`**: Implements the nodes of the search tree, which store information about the game state and search statistics.
- **`search.py`**: Contains the main MCTS search algorithm, which builds and explores the game tree.

### `neurals/`

This directory contains the neural network models that guide the MCTS search. The models are used to evaluate the value of game states and to predict the probability of different moves.

- **`models.py`**: Defines the architectures of the neural networks.
- **`network.py`**: Provides a wrapper for the neural network models, making them easy to use within the MCTS algorithm.

### `notebooks/`

This directory contains Jupyter notebooks for various tasks, such as:

- **`test_network.ipynb`**: For testing and debugging the neural network models.
- **`visualize_model.ipynb`**: For visualizing the learned representations of the game state.
- **`visualize_trees.ipynb`**: For visualizing the MCTS search trees, which can help in understanding the agent's decision-making process.

### `tests/`

This directory contains unit tests for the project, ensuring the correctness of the implementation. The tests cover:

- **`test_board.py`**: The game board and its logic.
- **`test_encoded.py`**: The encoding of the game state.
- **`test_move.py`**: The application of moves.
- **`test_perf_utils.py`**: The performance of utility functions.

## Getting Started

To run the project, you will need to install the dependencies listed in `requirements.txt`. You can then use the scripts in the root directory to train and evaluate the agent.

- **`manuals_control.py`**: Allows you to play the game manually.
- **`evaluate.py`**: Evaluates the performance of a trained agent.

## Conclusion

This project provides a complete implementation of a sophisticated AI agent for the game of 2048, based on a state-of-the-art planning algorithm. The modular structure of the codebase makes it easy to understand, modify, and extend, providing a solid foundation for further research in this area.
