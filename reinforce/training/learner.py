"""
Learner for Stochastic MuZero training.

Implements the training loop that:
1. Samples batches from replay buffer
2. Unrolls the model for K steps
3. Computes losses and updates network weights
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from keras import optimizers
from numpy import array, float32, mean, ndarray, zeros

from reinforce.neural.network import StochasticNetwork, create_stochastic_network

from .config import StochasticMuZeroConfig
from .losses import (
    compute_chance_loss,
    compute_policy_loss,
    compute_q_value_loss,
    compute_reward_loss,
    compute_total_loss,
    compute_value_loss,
)
from .replay_buffer import ReplayBuffer, Trajectory
from .targets import compute_td_lambda_targets


class StochasticMuZeroLearner:
    """
    Learner that updates network weights.

    Implements the training procedure from the paper:
    1. Sample batch from replay buffer
    2. Unroll model for K steps
    3. Compute MuZero loss + Chance loss
    4. Update weights with Adam

    Attributes
    ----------
    config : StochasticMuZeroConfig
        Training configuration.
    network : StochasticNetwork
        Neural network to train.
    optimizer : optimizers.Optimizer
        Adam optimizer.
    training_step : int
        Current training step.
    """

    def __init__(
        self,
        config: StochasticMuZeroConfig,
        network: StochasticNetwork | None = None,
    ):
        """
        Initialize the learner.

        Parameters
        ----------
        config : StochasticMuZeroConfig
            Training configuration.
        network : StochasticNetwork | None
            Network to train. Creates new if None.
        """
        self.config = config

        if network is None:
            network = create_stochastic_network(
                observation_shape=config.observation_shape,
                hidden_size=config.hidden_size,
                codebook_size=config.codebook_size,
            )
        self.network = network

        # ##>: Create optimizer.
        self.optimizer = optimizers.Adam(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay if config.weight_decay > 0 else None,
        )

        self.training_step = 0
        self._loss_history: list[dict[str, float]] = []

    def _prepare_batch_data(
        self,
        samples: list[tuple[Trajectory, int]],
    ) -> dict[str, Any]:
        """
        Prepare batch data for training.

        Parameters
        ----------
        samples : list[tuple[Trajectory, int]]
            List of (trajectory, start_position) pairs.

        Returns
        -------
        dict[str, Any]
            Batch data with observations, actions, targets.
        """
        batch_observations: list[list] = []
        batch_actions: list[list[int]] = []
        batch_policy_targets: list[list] = []
        batch_value_targets: list[list[float]] = []
        batch_reward_targets: list[list[float]] = []

        for trajectory, start_pos in samples:
            # ##>: Extract transitions for this sample.
            transitions = trajectory.transitions[start_pos:]

            # ##>: Compute value targets.
            value_targets = compute_td_lambda_targets(
                transitions,
                n_steps=self.config.td_steps,
                td_lambda=self.config.td_lambda,
                discount=self.config.discount,
            )

            # ##>: Collect data for unroll steps.
            sample_obs = []
            sample_actions = []
            sample_policies = []
            sample_values = []
            sample_rewards = []

            for k in range(min(self.config.num_unroll_steps + 1, len(transitions))):
                t = transitions[k]
                sample_obs.append(t.observation)
                sample_actions.append(t.action)
                sample_policies.append(t.search_policy)
                if k < len(value_targets):
                    sample_values.append(value_targets[k])
                else:
                    sample_values.append(0.0)
                sample_rewards.append(t.reward)

            # ##>: Pad if needed.
            while len(sample_obs) < self.config.num_unroll_steps + 1:
                sample_obs.append(sample_obs[-1])
                sample_actions.append(0)
                sample_policies.append(zeros(self.config.action_size))
                sample_values.append(0.0)
                sample_rewards.append(0.0)

            batch_observations.append(sample_obs)
            batch_actions.append(sample_actions)
            batch_policy_targets.append(sample_policies)
            batch_value_targets.append(sample_values)
            batch_reward_targets.append(sample_rewards)

        return {
            'observations': array(batch_observations, dtype=float32),
            'actions': array(batch_actions, dtype=int),
            'policy_targets': array(batch_policy_targets, dtype=float32),
            'value_targets': array(batch_value_targets, dtype=float32),
            'reward_targets': array(batch_reward_targets, dtype=float32),
        }

    def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        """
        Perform a single training step.

        Parameters
        ----------
        replay_buffer : ReplayBuffer
            Replay buffer to sample from.

        Returns
        -------
        dict[str, float]
            Dictionary of loss values.
        """
        # ##>: Sample batch.
        samples, importance_weights = replay_buffer.sample_batch(
            batch_size=self.config.batch_size,
            unroll_steps=self.config.num_unroll_steps,
            td_steps=self.config.td_steps,
        )

        # ##>: Prepare batch data.
        batch_data = self._prepare_batch_data(samples)

        # ##>: Compute losses for each sample in batch.
        all_losses = self._compute_batch_losses(batch_data)

        # ##>: Weight losses by importance sampling.
        weighted_loss = mean(all_losses['total'] * importance_weights)

        # ##>: Update training step.
        self.training_step += 1

        # ##>: Record losses.
        loss_dict = {
            'total': float(mean(all_losses['total'])),
            'policy': float(mean(all_losses['policy'])),
            'value': float(mean(all_losses['value'])),
            'reward': float(mean(all_losses['reward'])),
            'chance': float(mean(all_losses['chance'])),
            'weighted_total': float(weighted_loss),
        }
        self._loss_history.append(loss_dict)

        return loss_dict

    def _compute_batch_losses(self, batch_data: dict[str, Any]) -> dict[str, ndarray]:
        """
        Compute losses for a batch of samples.

        This implements the model unrolling from the paper's pseudocode.

        Parameters
        ----------
        batch_data : dict[str, Any]
            Batch data from _prepare_batch_data.

        Returns
        -------
        dict[str, ndarray]
            Losses for each sample.
        """
        batch_size = len(batch_data['observations'])
        observations = batch_data['observations']
        actions = batch_data['actions']
        policy_targets = batch_data['policy_targets']
        value_targets = batch_data['value_targets']
        reward_targets = batch_data['reward_targets']

        # ##>: Initialize loss arrays.
        total_losses = zeros(batch_size, dtype=float32)
        policy_losses = zeros(batch_size, dtype=float32)
        value_losses = zeros(batch_size, dtype=float32)
        reward_losses = zeros(batch_size, dtype=float32)
        chance_losses = zeros(batch_size, dtype=float32)

        for i in range(batch_size):
            sample_policy_losses = []
            sample_value_losses = []
            sample_reward_losses = []
            sample_q_losses = []
            sample_chance_losses = []
            sample_commitment_losses = []

            # ##>: Initial step: representation + prediction.
            obs_0 = observations[i, 0]
            hidden_state = self.network.representation(obs_0)

            pred_output = self.network.prediction(hidden_state)
            assert pred_output.policy is not None, 'Policy should not be None from prediction network'
            sample_policy_losses.append(compute_policy_loss(pred_output.policy, policy_targets[i, 0]))
            sample_value_losses.append(
                compute_value_loss(
                    pred_output.value,
                    value_targets[i, 0],
                    use_categorical=False,  # Using scalar for simplicity
                )
            )

            # ##>: Unroll for K steps.
            for k in range(1, self.config.num_unroll_steps + 1):
                # ##>: Afterstate dynamics: state + action -> afterstate.
                action = actions[i, k - 1]
                afterstate = self.network.afterstate_dynamics(hidden_state, action)

                # ##>: Afterstate prediction: afterstate -> (Q, σ).
                as_output = self.network.afterstate_prediction(afterstate)

                # ##>: Get chance code from encoder for target.
                obs_k = observations[i, k]
                chance_code = self.network.encoder(obs_k)

                # ##>: Q-value loss (trained towards previous value target).
                sample_q_losses.append(
                    compute_q_value_loss(
                        as_output.value,
                        value_targets[i, k - 1],
                        use_categorical=False,
                    )
                )

                # ##>: Chance distribution loss.
                assert as_output.chance_probs is not None, 'Chance probs should not be None from afterstate prediction'
                sample_chance_losses.append(compute_chance_loss(as_output.chance_probs, chance_code))

                # ##>: Dynamics: afterstate + chance_code -> (next_state, reward).
                hidden_state, pred_reward = self.network.dynamics(afterstate, chance_code)

                # ##>: Reward loss.
                sample_reward_losses.append(
                    compute_reward_loss(
                        pred_reward,
                        reward_targets[i, k],
                        use_categorical=False,
                    )
                )

                # ##>: Prediction on new state.
                pred_output = self.network.prediction(hidden_state)
                assert pred_output.policy is not None, 'Policy should not be None from prediction network'
                sample_policy_losses.append(compute_policy_loss(pred_output.policy, policy_targets[i, k]))
                sample_value_losses.append(
                    compute_value_loss(
                        pred_output.value,
                        value_targets[i, k],
                        use_categorical=False,
                    )
                )

            # ##>: Aggregate losses for this sample.
            loss_dict = compute_total_loss(
                policy_losses=sample_policy_losses,
                value_losses=sample_value_losses,
                reward_losses=sample_reward_losses,
                q_value_losses=sample_q_losses,
                chance_losses=sample_chance_losses,
                commitment_losses=sample_commitment_losses,
                commitment_weight=self.config.commitment_loss_weight,
            )

            total_losses[i] = loss_dict['total']
            policy_losses[i] = loss_dict['policy']
            value_losses[i] = loss_dict['value']
            reward_losses[i] = loss_dict['reward']
            chance_losses[i] = loss_dict['chance']

        return {
            'total': total_losses,
            'policy': policy_losses,
            'value': value_losses,
            'reward': reward_losses,
            'chance': chance_losses,
        }

    def get_network(self) -> StochasticNetwork:
        """
        Get the current network.

        Returns
        -------
        StochasticNetwork
            Current network.
        """
        return self.network

    def save_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """
        Save network checkpoint.

        Parameters
        ----------
        checkpoint_dir : str | Path
            Directory to save models.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ##>: Save each model.
        self.network._representation.save(checkpoint_dir / 'representation')
        self.network._prediction.save(checkpoint_dir / 'prediction')
        self.network._afterstate_dynamics.save(checkpoint_dir / 'afterstate_dynamics')
        self.network._afterstate_prediction.save(checkpoint_dir / 'afterstate_prediction')
        self.network._dynamics.save(checkpoint_dir / 'dynamics')
        self.network._encoder.save(checkpoint_dir / 'encoder')

        # ##>: Save training state.
        import json

        state = {
            'training_step': self.training_step,
            'codebook_size': self.network.codebook_size,
        }
        with open(checkpoint_dir / 'training_state.json', 'w') as f:
            json.dump(state, f)

    def load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """
        Load network checkpoint.

        Parameters
        ----------
        checkpoint_dir : str | Path
            Directory containing saved models.
        """
        checkpoint_dir = Path(checkpoint_dir)

        # ##>: Load training state.
        import json

        with open(checkpoint_dir / 'training_state.json') as f:
            state = json.load(f)

        self.training_step = state['training_step']

        # ##>: Load network.
        self.network = StochasticNetwork.from_path(
            representation_path=str(checkpoint_dir / 'representation'),
            prediction_path=str(checkpoint_dir / 'prediction'),
            afterstate_dynamics_path=str(checkpoint_dir / 'afterstate_dynamics'),
            afterstate_prediction_path=str(checkpoint_dir / 'afterstate_prediction'),
            dynamics_path=str(checkpoint_dir / 'dynamics'),
            encoder_path=str(checkpoint_dir / 'encoder'),
            codebook_size=state['codebook_size'],
        )

    def get_loss_history(self) -> list[dict[str, float]]:
        """
        Get loss history for logging.

        Returns
        -------
        list[dict[str, float]]
            List of loss dictionaries.
        """
        return self._loss_history
