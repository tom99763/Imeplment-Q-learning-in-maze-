import numpy as np
import pandas as pd
import random
import time




class Q_learning:
    def __init__(self, actions=None, lr=0.001, discount=0.99, epsilon=0.1, decay=1e-6, pretrain=False):
        self.actions = actions

        self.discount = discount

        self.epsilon = epsilon

        # after visit state-action pair which havent been visited,remember it.
        if not pretrain:
            self.q_table = pd.DataFrame(columns=self.actions)

        self.lr = lr

        self.decay = decay

    # ε-greedy method

    def choose_action(self, observation):
        # check state in qtable,if not exist add into q table(remember it)
        self.check_state_exist(observation)

        # exploitation
        if random.random() > self.epsilon:
            state_action = self.q_table.xs(observation)

            # random
            idx = np.random.permutation(state_action.index)
            state_action = state_action.reindex(idx)
            action = idx[np.argmax(state_action)]
            # print('eploit')
        # exploration
        else:
            action = np.random.choice(self.actions)
            self.epsilon -= self.decay
            # print('explore')
        return action

    def learn(self, s, a, r, s_, t):
        # check state in qtable,if not exist add into q table(remember it)
        self.check_state_exist(s_)
        # get old Q(s,a)
        q_pred = self.q_table.xs(s)[a]

        # q_target=R(s,a)+gamma*V(s_,a)
        if t == False:
            q_target = r+self.discount*self.q_table.xs(s_).max()
        else:
            q_target = r  # s_ = terminal state

        # Q(s,a) update
        self.q_table.xs(s)[a] += self.lr*(q_target-q_pred)

    # check visiting
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # add new s
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )

    def load_pretrain_model(self, model_name):
        self.q_table = pd.read_csv(
            f'./result_model/{model_name}--Q_table.csv', index_col=0)
