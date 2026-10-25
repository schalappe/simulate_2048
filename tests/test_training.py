"""
Tests for the Stochastic MuZero training pipeline: value support, targets, replay buffer, losses, search,
self-play, checkpoints, and resume.
"""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reinforce.evaluate import evaluate
from reinforce.game.core import create_initial_board, encode_observation, legal_actions_mask
from reinforce.mcts.stochastic_mctx import run_mcts
from reinforce.neural.support import inverse_scale_value, logits_to_scalar, scalar_to_two_hot, scale_value
from reinforce.training.config import TrainConfig, tiny_config
from reinforce.training.learner import CheckpointManager, apply_fns_for, create_train_state, train_step
from reinforce.training.losses import compute_loss
from reinforce.training.replay_buffer import Game, ReplayBuffer, td_lambda_targets
from reinforce.training.self_play import GameCollector, new_actors, play_moves
from reinforce.training.trainer import Trainer


def micro_config(**overrides) -> TrainConfig:
    """Smallest configuration that exercises every code path quickly."""
    base = replace(
        tiny_config(),
        hidden_size=8,
        num_residual_blocks=1,
        num_simulations=4,
        num_parallel_games=2,
        moves_per_generation=16,
        generation_interval=2,
        sequence_length=8,
        num_unroll_steps=3,
        td_steps=3,
        min_buffer_size=2,
        batch_size=8,
        log_interval=2,
        checkpoint_interval=4,
        eval_interval=1000,
        eval_games=1,
    )
    return replace(base, **overrides)


def make_game(length: int, terminal: bool = True) -> Game:
    """A synthetic game whose board at position t holds exponent t + 1 in its first cell."""
    boards = np.zeros((length + 1, 16), dtype=np.uint8)
    boards[:, 0] = np.arange(1, length + 2)
    return Game(
        boards=boards,
        actions=(np.arange(length) % 4).astype(np.int8),
        rewards=np.arange(1, length + 1, dtype=np.float32),
        policies=np.tile(np.array([0.7, 0.1, 0.1, 0.1], dtype=np.float32), (length, 1)),
        search_values=np.full(length, 10.0, dtype=np.float32),
        terminal=terminal,
    )


class TestValueSupport:
    """Categorical value representation."""

    def test_inverse_round_trip(self):
        values = jnp.array([0.0, 1.0, 37.5, 1_000.0, 150_000.0])
        assert jnp.allclose(inverse_scale_value(scale_value(values)), values, rtol=1e-4, atol=1e-3)

    def test_two_hot_reads_back_as_scalar(self):
        values = jnp.array([0.0, 4.0, 523.0, 90_000.0])
        two_hot = scalar_to_two_hot(values)

        assert jnp.allclose(jnp.sum(two_hot, axis=-1), 1.0)
        expected_bin = jnp.sum(two_hot * jnp.arange(two_hot.shape[-1]), axis=-1)
        assert jnp.allclose(expected_bin, scale_value(values), atol=1e-4)
        # ##>: Near-one-hot logits on the two-hot target read back as the raw value.
        assert jnp.allclose(logits_to_scalar(jnp.log(two_hot + 1e-12)), values, rtol=1e-3, atol=1e-2)


class TestValueTargets:
    """TD(λ) targets, checked by hand with γ = 1, n = 2, λ = 0.5."""

    def test_terminal_game(self):
        rewards = np.array([1.0, 2.0, 3.0])
        values = np.array([10.0, 20.0, 30.0])
        targets = td_lambda_targets(rewards, values, terminal=True, discount=1.0, td_steps=2, td_lambda=0.5)
        # ##>: t=0: 0.5(1+20) + 0.5(1+2+30); t=1: 0.5(2+30) + 0.5(2+3); t=2: only the final reward.
        assert np.allclose(targets, [27.0, 18.5, 3.0])

    def test_truncated_game_bootstraps_from_last_searched_state(self):
        rewards = np.array([1.0, 2.0, 3.0])
        values = np.array([10.0, 20.0, 30.0])
        targets = td_lambda_targets(rewards, values, terminal=False, discount=1.0, td_steps=2, td_lambda=0.5)
        # ##>: The final board was never searched: returns stop at state 2 and never count its reward twice.
        # ##>: t=0: 0.5(1+20) + 0.5(1+2+30); t=1: both returns are 2+30; t=2: its own search value.
        assert np.allclose(targets, [27.0, 32.0, 30.0])


class TestReplayBuffer:
    """Sequence storage, absorbing targets, sampling, and persistence."""

    def test_game_split_into_sequences_with_tail(self):
        config = micro_config(sequence_length=4, num_unroll_steps=2)
        buffer = ReplayBuffer(config)
        buffer.add_game(make_game(10))

        assert len(buffer) == 3
        assert buffer.num_positions == 10
        # ##>: The first sequence carries the next K steps (positions 4 and 5) and the board after them.
        assert list(buffer._boards[0, :, 0]) == [1, 2, 3, 4, 5, 6, 7]
        assert buffer._transition_mask[0].all()

    def test_absorbing_targets_after_terminal(self):
        config = micro_config(sequence_length=4, num_unroll_steps=2)
        buffer = ReplayBuffer(config)
        buffer.add_game(make_game(3, terminal=True))

        assert list(buffer._policy_mask[0]) == [True, True, True, False, False, False, False]
        assert list(buffer._value_mask[0]) == [True] * 7
        assert np.all(buffer._values[0, 3:] == 0.0)
        assert list(buffer._transition_mask[0]) == [True, True, True, False, False, False]
        assert list(buffer._reward_mask[0]) == [True] * 6

    def test_unknown_targets_after_truncation_are_masked(self):
        config = micro_config(sequence_length=4, num_unroll_steps=2)
        buffer = ReplayBuffer(config)
        buffer.add_game(make_game(3, terminal=False))

        assert not buffer._value_mask[0, 3:].any()
        assert not buffer._reward_mask[0, 3:].any()

    def test_sample_shapes_and_weights(self):
        config = micro_config()
        buffer = ReplayBuffer(config)
        buffer.add_game(make_game(20))
        batch = buffer.sample(16)

        unroll = config.num_unroll_steps
        assert batch.boards.shape == (16, unroll + 1, 16)
        assert batch.actions.shape == (16, unroll)
        assert np.all(batch.weights > 0) and np.isclose(batch.weights.max(), 1.0)

    def test_save_load_round_trip(self, tmp_path: Path):
        config = micro_config()
        buffer = ReplayBuffer(config)
        buffer.add_game(make_game(20))
        buffer.save(tmp_path / 'buffer.npz')

        restored = ReplayBuffer(config)
        restored.load(tmp_path / 'buffer.npz')

        assert len(restored) == len(buffer)
        assert restored.games_added == 1
        assert np.array_equal(restored.sample(8).boards, buffer.sample(8).boards)

    def test_load_rejects_other_configuration(self, tmp_path: Path):
        buffer = ReplayBuffer(micro_config())
        buffer.save(tmp_path / 'buffer.npz')

        with pytest.raises(ValueError, match='does not match'):
            ReplayBuffer(micro_config(sequence_length=16)).load(tmp_path / 'buffer.npz')


class TestLosses:
    """Every model, including the encoder, learns from the loss."""

    def test_every_model_receives_gradient(self):
        config = micro_config()
        state = create_train_state(config, jax.random.PRNGKey(0))
        buffer = ReplayBuffer(config)
        buffer.add_game(make_game(20))
        batch = buffer.sample(config.batch_size)

        grads = jax.grad(lambda p: compute_loss(p, apply_fns_for(config), batch, config)[0])(state.params)

        for name, model_grads in grads._asdict().items():
            norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(model_grads)))
            assert norm > 0, f'{name} receives no gradient'

    def test_train_step_updates_encoder(self):
        config = micro_config()
        state = create_train_state(config, jax.random.PRNGKey(0))
        before = jax.tree.map(jnp.copy, state.params.encoder)
        buffer = ReplayBuffer(config)
        buffer.add_game(make_game(20))

        # ##>: The warmup starts at learning rate 0, so the first step never moves parameters.
        for _ in range(2):
            state, losses = train_step(state, buffer.sample(config.batch_size), config)

        changed = jax.tree.leaves(jax.tree.map(lambda a, b: bool(jnp.any(a != b)), before, state.params.encoder))
        assert any(changed)
        assert float(losses.commitment_loss) > 0
        assert int(state.step) == 2


class TestSearch:
    """Legal moves reach mctx."""

    def test_illegal_actions_get_no_visits(self):
        config = micro_config(num_simulations=16)
        params = create_train_state(config, jax.random.PRNGKey(0)).params
        # ##>: Only "down" (action 3) changes this board.
        board = jnp.array([[2, 4, 2, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        legal = legal_actions_mask(board)
        assert list(legal) == [False, False, False, True]

        output = run_mcts(
            params,
            apply_fns_for(config),
            encode_observation(board)[None],
            legal[None],
            jax.random.PRNGKey(1),
            codebook_size=config.codebook_size,
            num_simulations=config.num_simulations,
            discount=config.discount,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_fraction=config.dirichlet_fraction,
            pb_c_init=config.pb_c_init,
            pb_c_base=config.pb_c_base,
        )
        assert jnp.allclose(jnp.asarray(output.action_weights)[0], jnp.array([0.0, 0.0, 0.0, 1.0]))


class TestSelfPlay:
    """Compiled self-play and game collection."""

    def test_games_span_chunks_and_restart(self):
        config = micro_config(moves_per_generation=8)
        params = create_train_state(config, jax.random.PRNGKey(0)).params
        actors = new_actors(jax.random.PRNGKey(1), config.num_parallel_games)
        collector = GameCollector(config.num_parallel_games)
        games: list[Game] = []
        key = jax.random.PRNGKey(2)
        while len(games) < 2:
            key, chunk_key = jax.random.split(key)
            actors, records = play_moves(params, actors, chunk_key, 1.0, config, num_moves=config.moves_per_generation)
            games += [game for _, game in collector.add(records)]

        for game in games:
            assert len(game) > config.moves_per_generation
            assert game.terminal
            assert np.count_nonzero(game.boards[0]) == 2  # restarted from a fresh board
            assert np.all(game.policies.sum(axis=1) > 0.99)

    def test_safety_limit_truncates(self):
        config = micro_config(max_game_moves=5)
        params = create_train_state(config, jax.random.PRNGKey(0)).params
        actors = new_actors(jax.random.PRNGKey(1), config.num_parallel_games)
        _, records = play_moves(params, actors, jax.random.PRNGKey(2), 1.0, config, num_moves=12)
        games = [game for _, game in GameCollector(config.num_parallel_games).add(records)]

        assert len(games) == 4
        assert all(len(game) == 5 and not game.terminal for game in games)

    def test_initial_boards_are_fresh(self):
        boards = jax.vmap(create_initial_board)(jax.random.split(jax.random.PRNGKey(0), 3))
        assert jnp.all(jnp.sum(boards != 0, axis=(1, 2)) == 2)


class TestCheckpoints:
    """Resume keeps the full training state."""

    def test_restore_keeps_optimizer_state(self, tmp_path: Path):
        config = micro_config()
        state = create_train_state(config, jax.random.PRNGKey(0))
        buffer = ReplayBuffer(config)
        buffer.add_game(make_game(20))
        state, _ = train_step(state, buffer.sample(config.batch_size), config)

        manager = CheckpointManager(tmp_path / 'checkpoints')
        manager.save(state)
        manager.wait()
        restored = manager.restore(config)
        manager.close()

        assert restored is not None
        assert jax.tree.structure(restored) == jax.tree.structure(state)
        assert all(jax.tree.leaves(jax.tree.map(lambda a, b: bool(jnp.array_equal(a, b)), restored, state)))

    def test_trainer_resumes_run(self, tmp_path: Path):
        config = micro_config()
        trainer = Trainer(config, tmp_path)
        trainer.initialize()
        trainer.train(4)
        trainer.close()
        sequences = len(trainer.buffer)

        resumed = Trainer(config, tmp_path)
        resumed.initialize()
        assert resumed.step == 4
        assert len(resumed.buffer) == sequences
        assert (tmp_path / 'metrics.jsonl').read_text().count('"kind": "train"') == 2
        resumed.close()

        with pytest.raises(ValueError, match='different configuration'):
            Trainer(micro_config(hidden_size=16), tmp_path)

    def test_evaluate_loads_the_run(self, tmp_path: Path):
        config = micro_config()
        trainer = Trainer(config, tmp_path)
        trainer.initialize()
        trainer.train(4)
        trainer.close()

        result = evaluate(config, num_games=1, run_dir=tmp_path)

        assert result['step'] == 4
        assert result['games'] == 1
