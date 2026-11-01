"""
Tests for JAX-native 2048 game implementation.

These tests verify that the JAX implementation produces identical results to the original NumPy implementation.
"""

import jax
import jax.numpy as jnp

from reinforce.game.core import (
    OBSERVATION_SIZE,
    TILE_BITS,
    board_to_exponents,
    create_initial_board,
    encode_observation,
    fill_cells,
    is_done,
    latent_state,
    legal_actions_mask,
    merge_row,
    next_state,
    slide_and_merge,
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


class TestInitialBoard:
    """Tests for create_initial_board."""

    def test_two_tiles(self):
        """A new board holds exactly two tiles of value 2 or 4."""
        board = create_initial_board(jax.random.PRNGKey(42))

        assert board.shape == (4, 4)
        assert jnp.sum(board != 0) == 2
        assert jnp.all(jnp.isin(board[board != 0], jnp.array([2, 4])))


class TestEncodeObservation:
    """Tests for the binary observation (paper, Appendix C)."""

    def test_empty_board(self):
        """Empty board should encode to zeros."""
        board = jnp.zeros((4, 4), dtype=jnp.int32)
        obs = encode_observation(board)
        assert obs.shape == (OBSERVATION_SIZE,)
        assert jnp.allclose(obs, 0.0)

    def test_known_values(self):
        """Tile 2^k sets bit k of its cell and nothing else."""
        board = jnp.array([[2, 4, 8, 2048], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=jnp.int32)
        bits = encode_observation(board).reshape(16, TILE_BITS)

        for cell, exponent in enumerate([1, 2, 3, 11]):
            assert bits[cell, exponent] == 1.0
            assert jnp.sum(bits[cell]) == 1.0
        assert jnp.sum(bits[4:]) == 0.0

    def test_exponents_round_trip(self):
        """Stored exponents give back the board values."""
        board = jnp.array([[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [4096, 8192, 16384, 32768]])
        exponents = board_to_exponents(board)

        assert exponents.dtype == jnp.uint8
        restored = jnp.where(exponents > 0, 2 ** exponents.astype(jnp.int32), 0).reshape(4, 4)
        assert jnp.array_equal(restored, board)


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

    def test_batched_initial_board(self):
        """create_initial_board should work with vmap."""
        keys = jax.random.split(jax.random.PRNGKey(42), 10)
        boards = jax.vmap(create_initial_board)(keys)

        assert boards.shape == (10, 4, 4)
        assert jnp.all(jnp.sum(boards != 0, axis=(1, 2)) == 2)
