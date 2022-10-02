import simulate_2048
import gym

if __name__ == '__main__':
    env = gym.make('GameBoard')
    env.seed()

    print("New game:")
    env.reset()
    env.render()
    print("Start ...")

    done = False
    moves = 0
    while not done:
        action = env.np_random.choice(range(4), 1).item()
        next_state, reward, done, info = env.step(action)
        moves += 1

        print('Next Action: "{}"\n\nReward: {}'.format(simulate_2048.GameBoard.ACTIONS_STRING[action], reward))
        env.render()

    print('\nTotal Moves: {}'.format(moves))
