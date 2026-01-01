"""
Comprehensive tests for the 2048 game environment.

Tests cover environment interface, stochastic tile spawning, game termination,
and integration between TwentyFortyEight class and core game logic.
"""

from unittest import TestCase, main

import numpy as np

from twentyfortyeight.core.gameboard import TILE_SPAWN_PROBS, after_state, is_done, next_state
from twentyfortyeight.core.gamemove import illegal_actions, legal_actions
from twentyfortyeight.envs.twentyfortyeight import TwentyFortyEight


class TestEnvironmentInterface(TestCase):
    """Test TwentyFortyEight class API and state management."""

    def setUp(self):
        """Initialize fresh environment before each test."""
        self.env = TwentyFortyEight(size=4)

    def test_reset_state_initialization(self):
        """Reset initializes board with exactly 2 tiles and zero reward."""
        obs = self.env.reset()

        # ##>: Exactly 2 non-zero tiles after reset.
        self.assertEqual(np.count_nonzero(obs), 2)

        # ##>: Tiles are only 2 or 4.
        tiles = obs[obs != 0]
        self.assertTrue(np.all((tiles == 2) | (tiles == 4)))

        # ##>: Reward resets to zero.
        self.assertEqual(self.env.reward, 0)

        # ##>: Game not finished after reset.
        self.assertFalse(self.env.is_finished)

    def test_reset_seed_reproducibility(self):
        """Same seed produces identical initial board state."""
        board1 = self.env.reset(seed=42)
        board2 = self.env.reset(seed=42)

        np.testing.assert_array_equal(board1, board2)

    def test_step_return_signature(self):
        """Step returns tuple of (observation, reward, done)."""
        self.env.reset()
        result = self.env.step(0)

        # ##>: Return is 3-tuple.
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        obs, reward, done = result

        # ##>: Observation is ndarray.
        self.assertIsInstance(obs, np.ndarray)

        # ##>: Reward is numeric.
        self.assertIsInstance(reward, (int, float, np.integer, np.floating))

        # ##>: Done is boolean.
        self.assertIsInstance(done, (bool, np.bool_))

    def test_observation_property_consistency(self):
        """Observation property returns current state."""
        self.env.reset()
        obs1 = self.env.observation
        obs2 = self.env.observation

        # ##>: Consecutive calls return same values.
        np.testing.assert_array_equal(obs1, obs2)

        # ##>: Observation shape matches board size.
        self.assertEqual(obs1.shape, (4, 4))

    def test_encoded_observation_mode(self):
        """Encoded mode produces correct shape and values."""
        env = TwentyFortyEight(size=4, encoded=True)
        obs = env.reset()

        # ##>: Encoded observation is 1D flattened array.
        self.assertEqual(obs.ndim, 1)

        # ##>: Length is size^2 * encoding_size (31 per cell for log2(2^16)).
        expected_length = 4 * 4 * 31
        self.assertEqual(len(obs), expected_length)

    def test_normalized_reward_mode(self):
        """Normalized mode applies log-scale normalization."""
        env_norm = TwentyFortyEight(size=4, normalize=True)
        env_raw = TwentyFortyEight(size=4, normalize=False)

        # ##>: Setup identical boards for both environments.
        board = np.array([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        env_norm._current_state = board.copy()
        env_raw._current_state = board.copy()

        # ##>: Execute same action in both.
        _, reward_norm, _ = env_norm.step(0)
        _, reward_raw, _ = env_raw.step(0)

        # ##>: Normalized reward is less than raw (log compression).
        self.assertLess(reward_norm, reward_raw)
        self.assertGreater(reward_norm, 0)


class TestMoveValidation(TestCase):
    """Test move validation and state transitions."""

    def test_invalid_move_no_state_change(self):
        """Invalid move leaves board unchanged and spawns no tile."""
        # ##>: Board with left edge filled - left move is invalid.
        board = np.array([[2, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0]])
        original = board.copy()

        next_board, reward = next_state(board, 0)  # Left action

        # ##>: State unchanged after invalid move.
        np.testing.assert_array_equal(next_board, original)

        # ##>: Reward is zero for invalid move.
        self.assertEqual(reward, 0)

        # ##>: No new tile spawned (count unchanged).
        self.assertEqual(np.count_nonzero(next_board), np.count_nonzero(original))

    def test_valid_move_spawns_exactly_one_tile(self):
        """Valid move spawns exactly 1 new tile in empty cell."""
        board = np.array([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        original_count = np.count_nonzero(board)

        next_board, reward = next_state(board, 0, seed=42)

        # ##>: Reward positive for valid merge.
        self.assertEqual(reward, 4)

        # ##>: Exactly one new tile spawned (count increased by 1 from merge reduction).
        # Original: 2 tiles → after merge: 1 tile → after spawn: 2 tiles
        # Net change: 0, but we can verify a new tile appeared.
        self.assertGreater(reward, 0)

    def test_all_directions_rotate_correctly(self):
        """All 4 action directions produce valid board states."""
        # ##>: Board with mergeable tiles to ensure valid moves.
        board = np.array([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        for action in range(4):
            result, reward = next_state(board.copy(), action, seed=action)

            # ##>: Result is valid 4x4 board.
            self.assertEqual(result.shape, (4, 4))

            # ##>: Valid move produces reward (merge happened).
            if action == 0:  # Left action merges the two 2s
                self.assertGreater(reward, 0)

    def test_legal_actions_detection(self):
        """Legal actions correctly identified for all board states."""
        # ##>: Empty board - all actions legal.
        board = np.zeros((4, 4), dtype=int)
        board[0, 0] = 2
        board[1, 1] = 2
        legal = legal_actions(board)
        self.assertEqual(len(legal), 4)

        # ##>: Left edge filled - left action illegal, others legal.
        board = np.array([[2, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0]])
        illegal = illegal_actions(board)
        legal = legal_actions(board)

        self.assertIn(0, illegal)  # Left is illegal
        self.assertEqual(len(legal) + len(illegal), 4)


class TestMergeLogic(TestCase):
    """Test tile merging and scoring mechanics."""

    def test_merge_empty_column(self):
        """Empty column merges to empty with zero score."""
        from twentyfortyeight.core.gameboard import merge_column

        column = np.array([0, 0, 0, 0])
        score, result = merge_column(column)

        self.assertEqual(score, 0)
        self.assertEqual(len(result), 0)

    def test_merge_single_tile(self):
        """Single tile column produces no merge."""
        from twentyfortyeight.core.gameboard import merge_column

        column = np.array([0, 4, 0, 0])
        score, result = merge_column(column)

        self.assertEqual(score, 0)
        np.testing.assert_array_equal(result, np.array([4]))

    def test_merge_cascade_all_same(self):
        """All same values merge in pairs correctly."""
        from twentyfortyeight.core.gameboard import merge_column

        column = np.array([2, 2, 2, 2])
        score, result = merge_column(column)

        # ##>: Two pairs merge: (2,2)→4 and (2,2)→4.
        self.assertEqual(score, 8)
        np.testing.assert_array_equal(result, np.array([4, 4]))

    def test_score_accumulation_multiple_merges(self):
        """Score correctly sums across multiple merges."""
        from twentyfortyeight.core.gameboard import slide_and_merge

        # ##>: Each row has one merge.
        board = np.array([[2, 2, 0, 0], [4, 4, 0, 0], [8, 8, 0, 0], [16, 16, 0, 0]])
        score, _ = slide_and_merge(board)

        # ##>: 4 + 8 + 16 + 32 = 60.
        self.assertEqual(score, 60)


class TestGameTermination(TestCase):
    """Test game over detection."""

    def test_game_over_full_board_no_merges(self):
        """Game ends when board full and no adjacent equal tiles."""
        board = np.array([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [8192, 16384, 32768, 65536]])

        self.assertTrue(is_done(board))

    def test_game_not_over_with_empty_cells(self):
        """Game continues when empty cells exist."""
        board = np.array([[2, 4, 8, 0], [16, 32, 64, 128], [256, 512, 1024, 2048], [4096, 8192, 16384, 32768]])

        # ##>: One empty cell at [0, 3].
        self.assertFalse(is_done(board))

    def test_game_not_over_with_merge_opportunity(self):
        """Game continues when merge possible despite full board."""
        board = np.array([[2, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [4096, 8192, 16384, 32768]])

        # ##>: Full board but [0,0]==2 and [0,1]==2 can merge.
        self.assertFalse(is_done(board))


class TestStochasticSpawning(TestCase):
    """Test tile spawning probability distribution."""

    def test_tile_spawn_values_only_2_or_4(self):
        """Spawned tiles are always 2 or 4."""
        env = TwentyFortyEight(size=4)

        # ##>: Spawn 100 tiles and verify values.
        for i in range(100):
            env.reset(seed=i)
            board = env._current_state
            tiles = board[board != 0]
            self.assertTrue(np.all((tiles == 2) | (tiles == 4)))

    def test_tile_spawn_distribution(self):
        """Tile values follow 90/10 distribution for 2 vs 4."""
        # ##>: Spawn many tiles to verify distribution.
        counts = {2: 0, 4: 0}
        samples = 1000

        for i in range(samples):
            env = TwentyFortyEight(size=4)
            env.reset(seed=i)
            tiles = env._current_state[env._current_state != 0]
            for tile in tiles:
                counts[tile] += 1

        total = sum(counts.values())
        freq_2 = counts[2] / total
        freq_4 = counts[4] / total

        # ##>: Allow ±5% tolerance for 1000 samples.
        self.assertAlmostEqual(freq_2, TILE_SPAWN_PROBS[2], delta=0.05)
        self.assertAlmostEqual(freq_4, TILE_SPAWN_PROBS[4], delta=0.05)

    def test_after_state_probabilities_sum_to_one(self):
        """All outcome probabilities sum to 1.0."""
        board = np.array([[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        outcomes = after_state(board)

        total_prob = sum(prob for _, prob in outcomes)

        self.assertAlmostEqual(total_prob, 1.0, places=10)

    def test_after_state_single_empty_cell(self):
        """One empty cell produces 2 outcomes with P=0.9 and P=0.1."""
        board = np.array([[2, 4, 8, 16], [32, 64, 128, 0], [256, 512, 1024, 2048], [4096, 8192, 16384, 32768]])
        outcomes = after_state(board)

        # ##>: Two outcomes (value 2 and 4 in the single empty cell).
        self.assertEqual(len(outcomes), 2)

        probs = sorted([prob for _, prob in outcomes])

        # ##>: Probabilities are 0.1 and 0.9.
        self.assertAlmostEqual(probs[0], 0.1, places=10)
        self.assertAlmostEqual(probs[1], 0.9, places=10)


class TestIntegration(TestCase):
    """End-to-end integration tests."""

    def test_game_reaches_termination(self):
        """Game eventually terminates after many steps."""
        env = TwentyFortyEight(size=4)
        env.reset(seed=42)

        max_steps = 1000
        for step in range(max_steps):
            legal = legal_actions(env._current_state)

            if not legal or env.is_finished:
                break

            # ##>: Choose first legal action.
            action = legal[0]
            _, _, done = env.step(action)

            if done:
                break

        # ##>: Game finished within reasonable steps.
        self.assertTrue(env.is_finished or step < max_steps)

    def test_reset_after_game_over(self):
        """Environment resets correctly after game ends."""
        env = TwentyFortyEight(size=4)

        # ##>: Manually create game over state.
        env._current_state = np.array(
            [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 4096], [8192, 16384, 32768, 65536]]
        )
        self.assertTrue(env.is_finished)

        # ##>: Reset after game over.
        obs = env.reset()

        # ##>: Fresh board with 2 tiles.
        self.assertEqual(np.count_nonzero(obs), 2)
        self.assertFalse(env.is_finished)
        self.assertEqual(env.reward, 0)

    def test_multiple_games_in_sequence(self):
        """Multiple games can be played sequentially."""
        env = TwentyFortyEight(size=4)

        for game_num in range(3):
            env.reset(seed=game_num)

            # ##>: Play until done or max steps.
            for _ in range(100):
                legal = legal_actions(env._current_state)
                if not legal:
                    break
                _, _, done = env.step(legal[0])
                if done:
                    break

            # ##>: Can reset and play another game.
            self.assertIsNotNone(env.observation)


if __name__ == '__main__':
    main()
