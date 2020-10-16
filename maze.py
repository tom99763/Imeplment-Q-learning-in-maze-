import numpy as np

import gym
import gym_maze
import pandas as pd

from rl_brain import Q_learning


def simulate():

    env.render()
    
    total_reward_track=0
    for episode in range(num_episode):
        s = tuple(env.reset())
        total_reward = 0
        while True:
            #print('\r'+f'epsilon---{algorithm.epsilon}', end='', flush=True)

            action = algorithm.choose_action(s)

            s_, reward, t, info = env.step(ACTION[action])

            algorithm.learn(s,action,
                            reward, tuple(s_), t)

            s = tuple(s_)

            total_reward += reward

            if t:
                break

            if RENDER_MAZE:
                env.render()

        print(f'episode--{episode},total_reward---{total_reward}')
        
        #convergence condition
        if abs(total_reward_track-total_reward)<=1e-6:
            break
        total_reward_track=total_reward
    return algorithm.q_table


if __name__ == "__main__":
    game_name="maze-random-10x10-v0"
    num_episode = 300
    RENDER_MAZE = True

    env = gym.make(game_name)
    ACTION = env.actions

    algorithm = Q_learning(actions=np.arange(
        env.action_space.n), discount=0.99, lr=0.1, decay=1e-7, epsilon=0.2)

    Q_table = simulate()
    
    #save to csv,check Q distribution
    Q_table.to_csv(f'{game_name}---Q_table.csv')
