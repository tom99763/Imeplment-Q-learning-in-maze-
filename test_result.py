import numpy as np
import pandas as pd
import time
import gym
import gym_maze
from learning_model.rl_brain import Q_learning


def test():
    env.render()
    s = tuple(env.reset())
    while True:
        #print('\r'+f'epsilon---{algorithm.epsilon}', end='', flush=True)
        time.sleep(0.01)
        action = algorithm.choose_action(str(s))

        s_, reward, t, info = env.step(ACTION[int(action)])

        algorithm.learn(str(s), int(action),
                        reward, str(tuple(s_)), t)

        s = tuple(s_)
        time.sleep(0.01)
        if t:
            time.sleep(1)
            print('finish')
            break

        env.render()


if __name__ == "__main__":
    game_name = 'maze-sample-25x25-v0'
    env = gym.make(game_name)
    ACTION = env.actions
    algorithm = Q_learning(discount=0.99, lr=0.1, epsilon=0, pretrain=True)
    algorithm.load_pretrain_model(game_name)
    test()
