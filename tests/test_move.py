from unittest import TestCase, main

from numpy import array

from twentyfortyeight.core.gamemove import illegal_actions, legal_actions


class TestGameMove(TestCase):
    def test_illegal_actions(self):
        """
        Test if illegal actions are correctly identified.
        """
        board = array([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        illegal = illegal_actions(board)
        self.assertEqual(set(illegal), {0})

    def test_legal_actions(self):
        """
        Test if illegal actions are correctly identified.
        """
        board = array([[2, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        legal = legal_actions(board)
        self.assertEqual(set(legal), {1, 2, 3})


if __name__ == '__main__':
    main()
