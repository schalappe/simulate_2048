# -*- coding: utf-8 -*-
"""
DQN implementation for 2048 game using TensorFlow and Keras.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Tuple

import tensorflow as tf
from keras import Model, Optimizer, losses, models, ops, optimizers
from numpy import ndarray

from neurals import build_model_with_identity_blocks
from replay import ReplayBuffer, PrioritizedReplayBuffer
from simulate.wrappers import EncodedTwentyFortyEight, normalize_reward
from tqdm import tqdm

from dqn.agent import DQNActor


@dataclass(frozen=True)
class DQNConfig:
    """
    Configuration class for Deep Q-Network (DQN) training.

    Attributes:
    -----------
    batch_size : int
        The number of samples per gradient update. Default is 64.
    gamma : float
        Discount factor for future rewards. Default is 0.95.
    epsilon_start : float
        Initial exploration rate. Default is 1.0.
    epsilon_end : float
        Final exploration rate. Default is 0.1.
    epsilon_decay : float
        Rate at which epsilon decays. Default is 0.995.
    learning_rate : float
        Learning rate for the optimizer. Default is 0.001.
    """

    batch_size: int = field(default=64, metadata={"min": 1, "max": 256})
    gamma: float = field(default=0.95, metadata={"min": 0.0, "max": 1.0})
    epsilon_start: float = field(default=1.0, metadata={"min": 0.0, "max": 1.0})
    epsilon_end: float = field(default=0.1, metadata={"min": 0.0, "max": 1.0})
    epsilon_decay: int = field(default=10000)
    learning_rate: float = field(default=0.001, metadata={"min": 1e-6, "max": 1.0})

    def __post_init__(self):
        if self.epsilon_start < self.epsilon_end:
            raise ValueError("`epsilon_start` must be greater than or equal to `epsilon_end`.")

    def next_epsilon(self, epsilon: float) -> float:
        """
        Returns the next epsilon value based on the current epsilon and decay rate.

        Parameters
        ----------
        epsilon : float
            The current epsilon value.

        Returns
        -------
        float
            The next epsilon value.
        """
        new_epsilon = epsilon - (self.epsilon_start - self.epsilon_end) / 10000
        return max(self.epsilon_end, new_epsilon)


def build_models() -> Tuple[Model, Model]:
    """
    Build the main and target models for DQN.

    Returns
    -------
    Tuple[Model, Model]
        A tuple containing the main model and target model.
    """
    model = build_model_with_identity_blocks(state_size=496, num_blocks=5)
    target_model = models.clone_model(model)
    target_model.set_weights(model.get_weights())
    return model, target_model


def setup_training(config: DQNConfig, model: Model) -> Tuple:
    """
    Set up the training components for DQN.

    Parameters
    ----------
    config : DQNConfig
        Configuration for the DQN training.
    model : Model
        The main model for the DQN.


    Returns
    -------
    Tuple
        A tuple containing the agent, replay buffer, optimizer, and loss function.
    """
    agent = DQNActor(model, epsilon=config.epsilon_start, encodage_size=31)
    replay_buffer = PrioritizedReplayBuffer(capacity=10000)
    optimizer = optimizers.Adam(learning_rate=config.learning_rate)
    loss_fn = losses.Huber(reduction="none")
    return agent, replay_buffer, optimizer, loss_fn


def collect_experience(
    env: EncodedTwentyFortyEight, agent: DQNActor, replay_buffer: ReplayBuffer
) -> Tuple[ndarray, bool]:
    """
    Collect a single step of experience from the environment.

    Parameters
    ----------
    env : EncodedTwentyFortyEight
        The environment to interact with.
    agent : DQNActor
        The agent making decisions.
    replay_buffer : ReplayBuffer
        Buffer to store experiences.

    Returns
    -------
    tuple:
        A tuple containing the next state and done flag.
    """
    state = env.observation
    action = agent.choose_action(state)
    next_state, reward, done = env.step(action)
    replay_buffer.add(state, action, normalize_reward(reward), next_state, done)
    return next_state, done


def update_model(
    model: Model,
    target_model: Model,
    replay_buffer: PrioritizedReplayBuffer,
    optimizer: Optimizer,
    loss_fn: Callable,
    config: DQNConfig,
):
    """
    Perform a single update step on the model.

    Parameters
    ----------
    model : Model
        The main model to update.
    target_model : Model
        The target model for stable Q-value estimates.
    replay_buffer : PrioritizedReplayBuffer
        Buffer containing experiences.
    optimizer : Optimizer
        The optimizer for model updates.
    loss_fn : Callable
        The loss function for computing gradients.
    config : DQNConfig
        Configuration for the update.
    """
    # ##: Sample a batch of experiences from the buffer.
    states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(config.batch_size)
    dones = ops.cast(dones, dtype="float32")
    weights = ops.cast(weights, dtype="float32")

    # ##: Compute the current Q values for each experience in the batch.
    next_q_values = target_model(next_states)
    target_q_values = rewards + (1 - dones) * config.gamma * ops.amax(next_q_values, axis=1)

    # ##: Compute the loss between the predicted and target Q values.
    with tf.GradientTape() as tape:
        q_values = model(states)
        one_hot_actions = ops.one_hot(actions, 4)
        current_q_values = ops.sum(ops.multiply(q_values, one_hot_actions), axis=1)
        td_errors = target_q_values - current_q_values
        loss = ops.mean(weights * loss_fn(target_q_values, current_q_values))

    # ##: Compute the gradients of the loss with respect to each parameter in the model.
    # ##: Update the parameters using the optimizer and the gradients.
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # ##: Update priorities in the replay buffer.
    replay_buffer.update_priorities(indices, abs(td_errors.numpy()))


def train_dqn(config: DQNConfig, episodes: int, storage_path: Path) -> str:
    """
    Train a DQN agent on the 2048 game.

    Parameters
    ----------
    config : DQNConfig
        Configuration for the DQN.
    episodes : int
        Number of episodes to train for.
    storage_path : Path
        Path to store the trained model.

    Returns
    -------
    str
        Name of the saved model file.
    """
    # ##: Set up the necessary components for training the DQN.
    model, target_model = build_models()
    agent, replay_buffer, optimizer, loss_fn = setup_training(config, model)

    env = EncodedTwentyFortyEight()
    move_count = 0
    for _ in tqdm(range(episodes), desc="Train Deep Q-Network"):
        env.reset()
        done = False
        while not done:
            # ##: Play a single move and save it.
            _, done = collect_experience(env, agent, replay_buffer)
            move_count += 1

            #  ##: Train the model every 50 moves.
            if len(replay_buffer) > config.batch_size and move_count % 50 == 0:
                update_model(model, target_model, replay_buffer, optimizer, loss_fn, config)

            # ##: Update the epsilon and the target network.
            agent.epsilon = config.next_epsilon(agent.epsilon)
            if move_count % 5000 == 0:
                target_model.set_weights(model.get_weights())

    # ##: Save the trained model.
    model_name = "deep_q_prioritized_identity.keras"
    model.save(storage_path / model_name)
    return model_name


if __name__ == "__main__":
    train_dqn(
        config=DQNConfig(batch_size=128),
        episodes=1000,
        storage_path=Path.cwd().parent / "zoo",
    )
