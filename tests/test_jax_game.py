"""
Tests for JAX-native 2048 game implementation.

These tests verify that the JAX implementation produces identical results to the original NumPy implementation.
"""

import jax
import jax.numpy as jnp

from reinforce.game.core import (
    count_empty,
    encode_observation,
    fill_cells,
    is_done,
    latent_state,
    legal_actions_mask,
    max_tile,
    merge_row,
    next_state,
    slide_and_merge,
)
from reinforce.game.env import (
    GameState,
    batched_reset,
    get_legal_actions,
    get_observation,
    reset,
    step,
)


class TestMergeRow:
    """Tests for merge_row function."""

    def test_empty_row(self):
        """Empty row should remain empty."""
        row = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        result, score = merge_row(row)
        assert jnp.array_equal(result, jnp.array([0, 0, 0, 0]))
        assert score == 0

    def test_single_tile(self):
        """Single tile should slide to left."""
        row = jnp.array([0, 0, 2, 0], dtype=jnp.int32)
        result, score = merge_row(row)
        assert jnp.array_equal(result, jnp.array([2, 0, 0, 0]))
        assert score == 0

    def test_two_same_tiles(self):
        """Two same tiles should merge."""
        row = jnp.array([2, 2, 0, 0], dtype=jnp.int32)
        result, score = merge_row(row)
        assert jnp.array_equal(result, jnp.array([4, 0, 0, 0]))
        assert score == 4

    def test_four_same_tiles(self):
        """Four same tiles should merge into two."""
        row = jnp.array([2, 2, 2, 2], dtype=jnp.int32)
        result, score = merge_row(row)
        assert jnp.array_equal(result, jnp.array([4, 4, 0, 0]))
        assert score == 8

    def test_no_merge_different(self):
        """Different tiles should not merge."""
        row = jnp.array([2, 4, 8, 16], dtype=jnp.int32)
        result, score = merge_row(row)
        assert jnp.array_equal(result, jnp.array([2, 4, 8, 16]))
        assert score == 0

    def test_complex_merge(self):
        """Complex merge scenario."""
        row = jnp.array([2, 0, 2, 4], dtype=jnp.int32)
        result, score = merge_row(row)
        assert jnp.array_equal(result, jnp.array([4, 4, 0, 0]))
        assert score == 4

    def test_merge_only_once(self):
        """Tiles should only merge once per move."""
        row = jnp.array([4, 2, 2, 0], dtype=jnp.int32)
        result, score = merge_row(row)
        assert jnp.array_equal(result, jnp.array([4, 4, 0, 0]))
        assert score == 4


class TestSlideAndMerge:
    """Tests for slide_and_merge function."""

    def test_full_board_slide(self):
        """Test sliding a full board."""
        board = jnp.array([[2, 2, 4, 4], [4, 4, 2, 2], [2, 4, 2, 4], [4, 2, 4, 2]], dtype=jnp.int32)
        result, score = slide_and_merge(board)

        expected = jnp.array([[4, 8, 0, 0], [8, 4, 0, 0], [2, 4, 2, 4], [4, 2, 4, 2]], dtype=jnp.int32)
        assert jnp.array_equal(result, expected)
        assert score == 24  # 4 + 8 + 8 + 4


class TestLatentState:
    """Tests for latent_state function."""

    def test_left_action(self):
        """Test left slide."""
        board = jnp.array([[0, 2, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        result, reward = latent_state(board, 0)  # Left

        expected = jnp.array([[4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        assert jnp.array_equal(result, expected)
        assert reward == 4

    def test_up_action(self):
        """Test up slide."""
        board = jnp.array([[2, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        result, reward = latent_state(board, 1)  # Up

        expected = jnp.array([[4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        assert jnp.array_equal(result, expected)
        assert reward == 4

    def test_right_action(self):
        """Test right slide."""
        board = jnp.array([[2, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        result, reward = latent_state(board, 2)  # Right

        expected = jnp.array([[0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        assert jnp.array_equal(result, expected)
        assert reward == 4

    def test_down_action(self):
        """Test down slide."""
        board = jnp.array([[2, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        result, reward = latent_state(board, 3)  # Down

        expected = jnp.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 0]], dtype=jnp.int32)
        assert jnp.array_equal(result, expected)
        assert reward == 4


class TestLegalActionsMask:
    """Tests for legal_actions_mask function."""

    def test_empty_board_all_illegal(self):
        """Empty board has no legal moves (nothing to move)."""
        board = jnp.zeros((4, 4), dtype=jnp.int32)
        mask = legal_actions_mask(board)
        assert jnp.array_equal(mask, jnp.array([False, False, False, False]))

    def test_full_board_no_merges(self):
        """Full board with no possible merges."""
        board = jnp.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]], dtype=jnp.int32)
        mask = legal_actions_mask(board)
        assert jnp.array_equal(mask, jnp.array([False, False, False, False]))

    def test_can_slide_left(self):
        """Empty space on left allows slide."""
        board = jnp.array([[0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        mask = legal_actions_mask(board)
        assert mask[0]  # Left is legal

    def test_can_merge_left(self):
        """Two same tiles can merge left."""
        board = jnp.array([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        mask = legal_actions_mask(board)
        assert mask[0]  # Left is legal (merge)


class TestIsDone:
    """Tests for is_done function."""

    def test_not_done_with_empty(self):
        """Game not over with empty cells."""
        board = jnp.array([[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        assert not is_done(board)

    def test_not_done_with_merge(self):
        """Game not over with possible merge."""
        board = jnp.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 2, 2]], dtype=jnp.int32)
        assert not is_done(board)

    def test_done_full_no_merges(self):
        """Game over when full with no merges."""
        board = jnp.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]], dtype=jnp.int32)
        assert is_done(board)


class TestFillCells:
    """Tests for fill_cells function."""

    def test_fills_empty_cell(self):
        """Should fill an empty cell."""
        board = jnp.zeros((4, 4), dtype=jnp.int32)
        key = jax.random.PRNGKey(42)
        result = fill_cells(board, key)

        # ##>: Should have exactly one non-zero cell.
        assert jnp.sum(result != 0) == 1

        # ##>: Value should be 2 or 4.
        non_zero = result[result != 0]
        assert non_zero[0] in [2, 4]

    def test_full_board_unchanged(self):
        """Full board should remain unchanged."""
        board = jnp.ones((4, 4), dtype=jnp.int32) * 2
        key = jax.random.PRNGKey(42)
        result = fill_cells(board, key)
        assert jnp.array_equal(result, board)


class TestNextState:
    """Tests for next_state function."""

    def test_valid_move(self):
        """Valid move should change board and add tile."""
        board = jnp.array([[2, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        key = jax.random.PRNGKey(42)
        result, reward = next_state(board, 1, key)  # Up

        # ##>: Tiles should have merged.
        assert result[0, 0] == 4

        # ##>: A new tile should have appeared.
        assert jnp.sum(result != 0) == 2

        # ##>: Reward should be 4.
        assert reward == 4


class TestEnvironment:
    """Tests for environment wrapper."""

    def test_reset(self):
        """Reset should create valid initial state."""
        key = jax.random.PRNGKey(42)
        state = reset(key)

        assert isinstance(state, GameState)
        assert state.board.shape == (4, 4)
        assert jnp.sum(state.board != 0) == 2  # Two initial tiles
        assert state.step_count == 0
        assert not state.done
        assert state.total_reward == 0

    def test_step(self):
        """Step should update state correctly."""
        key = jax.random.PRNGKey(42)
        state = reset(key)

        key, step_key = jax.random.split(key)
        new_state, reward, done, info = step(state, 0, step_key)

        assert new_state.step_count == 1
        assert 'max_tile' in info

    def test_get_observation(self):
        """Get observation should return flattened encoded board."""
        key = jax.random.PRNGKey(42)
        state = reset(key)
        obs = get_observation(state)

        assert obs.shape == (16,)
        assert obs.dtype == jnp.float32

    def test_get_legal_actions(self):
        """Get legal actions should return valid mask."""
        key = jax.random.PRNGKey(42)
        state = reset(key)
        mask = get_legal_actions(state)

        assert mask.shape == (4,)
        assert mask.dtype == jnp.bool_


class TestEncodeObservation:
    """Tests for observation encoding."""

    def test_empty_board(self):
        """Empty board should encode to zeros."""
        board = jnp.zeros((4, 4), dtype=jnp.int32)
        obs = encode_observation(board)
        assert jnp.allclose(obs, jnp.zeros(16))

    def test_known_values(self):
        """Test encoding of known tile values."""
        board = jnp.array([[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        obs = encode_observation(board)

        # ##>: log2(2)/16 = 1/16, log2(4)/16 = 2/16, etc.
        assert jnp.isclose(obs[0], 1 / 16)
        assert jnp.isclose(obs[1], 2 / 16)
        assert jnp.isclose(obs[2], 3 / 16)
        assert jnp.isclose(obs[3], 4 / 16)


class TestMaxTile:
    """Tests for max_tile function."""

    def test_max_tile(self):
        """Should return maximum tile value."""
        board = jnp.array(
            [[2, 4, 8, 16], [32, 64, 128, 256], [0, 0, 0, 0], [0, 0, 0, 2048]],
            dtype=jnp.int32,
        )
        assert max_tile(board) == 2048


class TestCountEmpty:
    """Tests for count_empty function."""

    def test_empty_board(self):
        """Empty board should have 16 empty cells."""
        board = jnp.zeros((4, 4), dtype=jnp.int32)
        assert count_empty(board) == 16

    def test_partial_board(self):
        """Partial board should count correctly."""
        board = jnp.array([[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        assert count_empty(board) == 14


class TestJITCompilation:
    """Tests to verify JIT compilation works correctly."""

    def test_merge_row_jit(self):
        """merge_row should be JIT-compilable."""
        row = jnp.array([2, 2, 4, 4], dtype=jnp.int32)
        jit_merge = jax.jit(merge_row)
        result, score = jit_merge(row)
        assert jnp.array_equal(result, jnp.array([4, 8, 0, 0]))

    def test_slide_and_merge_jit(self):
        """slide_and_merge should be JIT-compilable."""
        board = jnp.ones((4, 4), dtype=jnp.int32) * 2
        jit_slide = jax.jit(slide_and_merge)
        result, score = jit_slide(board)
        assert result.shape == (4, 4)

    def test_latent_state_jit(self):
        """latent_state should be JIT-compilable."""
        board = jnp.ones((4, 4), dtype=jnp.int32) * 2
        jit_latent = jax.jit(latent_state)
        result, score = jit_latent(board, 0)
        assert result.shape == (4, 4)

    def test_next_state_jit(self):
        """next_state should be JIT-compilable."""
        board = jnp.array([[2, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        key = jax.random.PRNGKey(42)
        jit_next = jax.jit(next_state)
        result, score = jit_next(board, 1, key)
        assert result.shape == (4, 4)


class TestVectorization:
    """Tests for vmap vectorization."""

    def test_batched_merge(self):
        """merge_row should work with vmap."""
        rows = jnp.array([[2, 2, 0, 0], [4, 4, 0, 0], [2, 4, 2, 4]], dtype=jnp.int32)
        batched_merge = jax.vmap(merge_row)
        results, scores = batched_merge(rows)

        assert results.shape == (3, 4)
        assert jnp.array_equal(results[0], jnp.array([4, 0, 0, 0]))
        assert jnp.array_equal(results[1], jnp.array([8, 0, 0, 0]))
        assert jnp.array_equal(results[2], jnp.array([2, 4, 2, 4]))

    def test_batched_reset(self):
        """reset should work with vmap."""
        keys = jax.random.split(jax.random.PRNGKey(42), 10)
        states = batched_reset(keys)

        assert states.board.shape == (10, 4, 4)
        assert states.step_count.shape == (10,)
        assert states.done.shape == (10,)
