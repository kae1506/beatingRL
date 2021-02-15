import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
from main import SnakeEnv
import math
import time
import numpy as np
from npreplaymemory import ReplayBuffer
import matplotlib.pyplot as plt
from utils import plotLearning

np.random.seed(0)

class Network(nn.Module):
    def __init__(self, alpha, inputShape, numActions):
        super().__init__()
        self.inputShape = inputShape
        self.numActions = numActions
        self.fc1Dims = 1024
        self.fc2Dims = 512

        self.fc1 = nn.Linear(*self.inputShape, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)
        self.advantage = nn.Linear(self.fc2Dims, numActions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        # self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        a = self.advantage(x)

        return a

class Agent():
    def __init__(self, lr, input_shape, n_actions, eps_dec=0.001, eps_min=0.001):
        self.lr = lr
        self.gamma = 0.99
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.surprise = 0.5

        self.learn_cntr = 0
        self.replace = 100

        self.eps = 1
        self.eps_dec = eps_dec
        self.eps_min = eps_min

        self.model = Network(lr, self.input_shape, self.n_actions)
        self.target = Network(lr, self.input_shape, self.n_actions)
        self.memory = ReplayBuffer(100000, self.input_shape)


    def choose_action(self, state):
        if np.random.random() > self.eps:
            state = T.Tensor(state).to(self.model.device)
            states = state.unsqueeze(0)
            actions = self.model(states)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_ntwrk(self):
        self.target.load_state_dict(self.model.state_dict())

    def save_checkpoint(self, filename, memoryfile):
        if filename == None:
            filename = "deuling/DDDQN.weights"

        save_dict = {'policy': self.model.state_dict(),
             'optim': self.model.optimizer.state_dict(),
             'target': self.target.state_dict()
        }

        T.save(save_dict, filename)
        self.memory.save(memoryfile)

    def load_checkpoint(self, filename, memoryfile):
        load_dict = T.load(filename)

        self.model.load_state_dict(load_dict['policy'])
        self.model.optimizer.load_state_dict(load_dict['optim'])
        self.target.load_state_dict(load_dict['target'])

        self.model.train()
        self.target.train()
        self.memory.load(memoryfile)

    def learn(self, batchSize):
        if self.memory.memCount < batchSize:
            return

        self.model.optimizer.zero_grad()

        if self.learn_cntr % self.replace == 0:
            self.replace_ntwrk()

        state, action, reward, state_, done = self.memory.sample(batchSize)

        states  = T.Tensor(state).to(T.float32 ).to(self.model.device)
        actions = T.Tensor(action).to(T.int64   ).to(self.model.device)
        rewards = T.Tensor(reward).to(T.float32 ).to(self.model.device)
        states_ = T.Tensor(state_).to(T.float32 ).to(self.model.device)
        dones   = T.Tensor(done).to(T.bool    ).to(self.model.device)

        batch_indices = np.arange(batchSize, dtype=np.int64)
        qValue = self.model(states)
        qValue_ = self.target(states_)

        qValue_[dones] = 0.0

        td = rewards + self.gamma * qValue_
        loss = self.model.loss(td, qValue).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()

        self.eps -= self.eps_dec
        if self.eps < self.eps_min:
            self.eps = self.eps_min

        self.learn_cntr += 1


if __name__ == '__main__':
    BATCH_SIZE = 64
    FILENAME = "DQN.weights"
    MEMORYFILENAME = "MEMORY_DQN.mem"
    load = False #True
    n_games = 1000
    env = SnakeEnv(frameRate=15)
    agent = Agent(lr=0.001, input_shape=(2, 400), n_actions=5)
    memory = []
    scores = []
    Avg_scores = []
    highscore = -math.inf

    start_time =  time.process_time()

    if load:
        agent.load_checkpoint(FILENAME, MEMORYFILENAME)

    print(agent.memory.memCount)
    for i in range(n_games):
        state = env.reset()
        done=False

        score = 0
        frame = 0
        while not done:
            action = agent.choose_action(state)
            reward, done, state_ = env.step(action)
            agent.memory.storeMemory(state, action, reward, state_, done)
            agent.learn(BATCH_SIZE)

            score += reward
            frame += 1
            state = state_

        scores.append(score)
        highscore = max(highscore, score)

        print(( "ep {}: high-score {:12.3f}, "
                "score {:12.3f}, last-episode-time {:4d}").format(
            i, highscore, score, frame))

        avg_score = np.mean(scores[-100:])
        Avg_scores.append(avg_score)

    print(time.process_time() - start_time)
    agent.save_checkpoint(FILENAME, MEMORYFILENAME)
    plotLearning([i for i in range(n_games)], Avg_scores, scores, "del.png")
    plt.show()
