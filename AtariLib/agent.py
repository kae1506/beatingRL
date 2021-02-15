import torch
import numpy as np
from model import ConvNet
from npreplaymemory import ReplayBuffer

class Agent(object):
    def __init__(self, input_shape, numActions, lr=0.001, gamma=0.96, 
                mem_size=10000, eps=1.0, eps_dec=0.001, eps_end=0.01):
        self.input_shape= input_shape
        self.numActions = numActions
        self.actionSpace = [i for i in range(self.numActions)]
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps
        self.eps_dec = eps_dec
        self.eps_end = eps_end
        self.evalfilename = r"models/eval.weights"
        self.targetfilename = r"models/target.weights"
        self.memoryfilename = r"models/memory.mems"
        self.learn_step_counter = 0
        self.replace = 100

        self.memory = ReplayBuffer(mem_size, self.input_shape)
        self.model = ConvNet(self.input_shape, self.numActions, self.evalfilename)
        self.target = ConvNet(self.input_shape, self.numActions, self.targetfilename)
        self.batchSize = 64

    def load(self):
        self.model.load_model()
        self.target.load_model()
        self.memory.load(self.memoryfilename)

    def save(self):
        self.model.save_model()
        self.target.save_model()
        self.memory.save(self.memoryfilename)

    def choose_action(self, state):
        state = torch.tensor(state).unsqueeze(0)
        if np.random.random() > self.epsilon:
            actions = self.model(state)
            #print("picked an action /n --------------")
            
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.actionSpace)

        return action

    def replace_target_network(self):
        self.target.load_state_dict(self.model.state_dict())

    def train(self):
        if self.memory.memCount < self.batchSize:
            return 

        if self.learn_step_counter % self.replace == 0:
            self.replace_target_network()

        self.model.optimizer.zero_grad()

        states, actions, rewards, states_, dones = self.memory.sample(self.batchSize)
    
        states  = torch.Tensor(states).to(torch.float32 ).to(self.model.device)
        actions = torch.Tensor(actions).to(torch.int64   ).to(self.model.device)
        rewards = torch.Tensor(rewards).to(torch.float32 ).to(self.model.device)
        states_ = torch.Tensor(states_).to(torch.float32 ).to(self.model.device)
        dones   = torch.Tensor(dones).to(torch.bool    ).to(self.model.device)

        batch_indices = np.arange(self.batchSize, dtype=np.int64)
        q_eval = self.model(states)[batch_indices, actions]
        q_next = self.target(states_).max(dim=1)[0]
        q_next[dones] = 0.0

        #print("it was training /n ________")

        q_target = rewards + self.gamma*q_next
        loss = self.model.loss(q_target, q_eval).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()

        self.epsilon -= self.eps_dec
        if self.epsilon <= self.eps_end:
            self.epsilon = self.eps_end


        

