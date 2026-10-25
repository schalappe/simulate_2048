"""
Training loop for Stochastic MuZero: self-play, learning, logging, checkpoints, and resume.

A run directory holds everything a run needs to resume on another machine or a restarted pod:

    run_dir/
        config.json          configuration the run was started with
        checkpoints/         orbax checkpoints (parameters, optimizer state, step, key)
        replay_buffer.npz    replay buffer, saved with each checkpoint
        metrics.jsonl        one JSON object per log line ("train" and "eval" records)

Games in progress at the moment of a save are not stored; a resumed run starts new games.
"""

import json
import time
from pathlib import Path

import jax
import numpy as np

from reinforce.training.config import TrainConfig, load_config, save_config
from reinforce.training.learner import CheckpointManager, create_train_state, train_step
from reinforce.training.replay_buffer import Game, ReplayBuffer
from reinforce.training.self_play import (
    ActorState,
    GameCollector,
    evaluate_games,
    new_actors,
    play_moves,
    summarize_games,
)


class Trainer:
    """
    Run and resume training in a run directory.

    Parameters
    ----------
    config : TrainConfig
        Training configuration. A resumed run must use the configuration it was started with.
    run_dir : str | Path
        Run directory (see module docstring).

    Raises
    ------
    ValueError
        If `run_dir` holds a run started with a different configuration.
    """

    def __init__(self, config: TrainConfig, run_dir: str | Path):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        config_path = self.run_dir / 'config.json'
        if config_path.exists():
            saved = load_config(config_path)
            if saved != config:
                raise ValueError(
                    f'{self.run_dir} holds a run started with a different configuration. '
                    'Use the same --mode or a new --run-dir.'
                )
        else:
            save_config(config, config_path)

        self._checkpoints = CheckpointManager(self.run_dir / 'checkpoints')
        self._buffer_path = self.run_dir / 'replay_buffer.npz'
        self._metrics_path = self.run_dir / 'metrics.jsonl'
        self.buffer = ReplayBuffer(config, seed=config.seed)
        self.state = create_train_state(config, jax.random.PRNGKey(config.seed))
        self._actors: ActorState | None = None
        self._collector = GameCollector(config.num_parallel_games)
        self._recent_games: list[Game] = []
        self._self_play_moves = 0
        self._self_play_seconds = 0.0

    @property
    def step(self) -> int:
        """Return the number of training steps done."""
        return int(self.state.step)

    def initialize(self) -> None:
        """Resume from the run directory if it holds a checkpoint, otherwise keep the fresh state."""
        restored = self._checkpoints.restore(self.config)
        if restored is None:
            print('Initialized new training state at step 0')
        else:
            self.state = restored
            print(f'Restored training state from step {self.step}')
        if self._buffer_path.exists():
            self.buffer.load(self._buffer_path)
            print(f'Restored replay buffer: {len(self.buffer)} sequences, {self.buffer.num_positions} positions')
        key, actor_key = jax.random.split(self.state.key)
        self.state = self.state._replace(key=key)
        self._actors = new_actors(actor_key, self.config.num_parallel_games)

    def fill_buffer(self) -> None:
        """Play self-play chunks until the buffer holds `min_buffer_size` sequences."""
        while not self.buffer.is_ready():
            self._play_chunk(self.step)
            print(f'Filling buffer: {len(self.buffer)}/{self.config.min_buffer_size} sequences')

    def train(self, num_steps: int) -> dict:
        """
        Train for `num_steps` steps, then save. Ctrl-C stops early and still saves.

        Parameters
        ----------
        num_steps : int
            Number of training steps to run.

        Returns
        -------
        dict
            Steps run, elapsed seconds, and steps per second.
        """
        config = self.config
        self.fill_buffer()
        step = self.step
        end_step = step + num_steps
        start_time = time.time()
        log_time = start_time
        start_step = step
        try:
            # ##>: JAX dispatches train_step asynchronously, so sampling the next batch on the host overlaps
            # ##>: with the device computing the current step. Nothing in this loop may block on device
            # ##>: values except logging, saving, and self-play.
            while step < end_step:
                if step % config.generation_interval == 0:
                    self._play_chunk(step)
                self.state, losses = train_step(self.state, self.buffer.sample(config.batch_size), config)
                step += 1

                if step % config.log_interval == 0:
                    now = time.time()
                    self._log_train(step, losses, config.log_interval / (now - log_time))
                    log_time = now
                if step % config.checkpoint_interval == 0:
                    self.save()
                if step % config.eval_interval == 0:
                    self.evaluate()
        except KeyboardInterrupt:
            print(f'Interrupted at step {step}; saving.')
        self.save()
        self._checkpoints.wait()
        elapsed = time.time() - start_time
        steps_run = step - start_step
        return {
            'steps': steps_run,
            'seconds': elapsed,
            'steps_per_second': steps_run / elapsed if elapsed > 0 else 0.0,
        }

    def evaluate(self, num_games: int | None = None) -> dict:
        """
        Play greedy games without root noise, log them, and return their summary.

        Parameters
        ----------
        num_games : int | None
            Number of games; `config.eval_games` if None.

        Returns
        -------
        dict
            Output of `summarize_games`, plus the training step.
        """
        key, eval_key = jax.random.split(self.state.key)
        self.state = self.state._replace(key=key)
        games = evaluate_games(self.state.params, eval_key, self.config, num_games or self.config.eval_games)
        summary = {'step': self.step, **summarize_games(games)}
        self._write_metrics('eval', summary)
        print(
            f'[eval] step {summary["step"]}: mean score {summary["mean_score"]:.0f}, '
            f'best tile {summary["max_tile"]}, mean length {summary["mean_length"]:.0f}'
        )
        return summary

    def save(self) -> None:
        """Save the replay buffer, then start saving the checkpoint of the current step."""
        self.buffer.save(self._buffer_path)
        self._checkpoints.save(self.state)

    def close(self) -> None:
        """Wait for pending checkpoint saves."""
        self._checkpoints.close()

    def _play_chunk(self, step: int) -> None:
        """Play one self-play chunk at the temperature of `step` and store the games that ended in it."""
        if self._actors is None:
            raise RuntimeError('Trainer not initialized. Call initialize() first.')
        key, chunk_key = jax.random.split(self.state.key)
        self.state = self.state._replace(key=key)
        temperature = self.config.get_temperature(step)

        start = time.time()
        actors, records = play_moves(
            self.state.params,
            self._actors,
            chunk_key,
            temperature,
            self.config,
            num_moves=self.config.moves_per_generation,
        )
        # ##>: Advance the actors only once the collector holds the chunk, so an interrupt while waiting for
        # ##>: the device drops the whole chunk instead of leaving a gap inside games in progress.
        finished = self._collector.add(records)
        self._actors = actors
        for _, game in finished:
            self.buffer.add_game(game)
            self._recent_games.append(game)
        self._self_play_seconds += time.time() - start
        self._self_play_moves += self.config.moves_per_generation * self.config.num_parallel_games

    def _log_train(self, step: int, losses, steps_per_second: float) -> None:
        """Write a train record and print a one-line summary."""
        record = {
            'step': step,
            **{name: float(value) for name, value in losses._asdict().items()},
            'steps_per_second': steps_per_second,
            'self_play_moves_per_second': self._self_play_moves / max(self._self_play_seconds, 1e-9),
            'buffer_sequences': len(self.buffer),
            'buffer_positions': self.buffer.num_positions,
            'games_added': self.buffer.games_added,
            'temperature': self.config.get_temperature(step),
        }
        if self._recent_games:
            recent = summarize_games(self._recent_games)
            record['recent_games'] = recent['games']
            record['recent_mean_score'] = recent['mean_score']
            record['recent_max_tile'] = recent['max_tile']
        self._recent_games = []
        self._write_metrics('train', record)
        print(
            f'step {step}: loss {record["total_loss"]:.3f}, {steps_per_second:.1f} steps/s, '
            f'{record["self_play_moves_per_second"]:.0f} moves/s, buffer {record["buffer_sequences"]} seq, '
            f'games {record["games_added"]}'
        )

    def _write_metrics(self, kind: str, record: dict) -> None:
        """Append one record to metrics.jsonl."""
        line = json.dumps({'kind': kind, 'time': time.time(), **_to_builtin(record)})
        with self._metrics_path.open('a') as file:
            file.write(line + '\n')


def _to_builtin(record: dict) -> dict:
    """Convert NumPy scalars so the record serializes to JSON."""
    return {key: value.item() if isinstance(value, np.generic) else value for key, value in record.items()}
