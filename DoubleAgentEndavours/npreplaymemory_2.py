import numpy as np
import torch

class ReplayBuffer2():
    def __init__(self, maxSize, stateShape):
        self.memSize = maxSize
        self.differentiating = 0
        self.memCount = 0

        self.stateMemory        = np.zeros((self.memSize, *stateShape), dtype=np.float32)
        self.actionMemory       = np.zeros( self.memSize,               dtype=np.int64  )
        self.rewardMemory       = np.zeros( self.memSize,               dtype=np.float32)
        self.nextStateMemory    = np.zeros((self.memSize, *stateShape), dtype=np.float32)
        self.doneMemory         = np.zeros( self.memSize,               dtype=np.bool   )

    def storeMemory(self, state, action, reward, nextState, done):
        memIndex = self.memCount % self.memSize

        self.stateMemory[memIndex]      = state
        self.actionMemory[memIndex]     = action
        self.rewardMemory[memIndex]     = reward
        self.nextStateMemory[memIndex]  = nextState
        self.doneMemory[memIndex]       = done

        self.memCount += 1

    def save(self, filename):
        if filename==None:
            filename = 'memory.mems'

        save_dict = {
            'state': self.stateMemory,
            'action': self.actionMemory,
            'reward': self.rewardMemory,
            'nextState': self.nextStateMemory,
            'done': self.doneMemory,
            'mem': self.memCount
        }

        torch.save(save_dict, filename)

    def load(self, filename):
        print(filename)
        load_dict = torch.load(filename)

        self.stateMemory     = load_dict['state']
        self.actionMemory    = load_dict['action']
        self.rewardMemory    = load_dict['reward']
        self.nextStateMemory = load_dict['nextState']
        self.doneMemory      = load_dict['done']
        self.memCount        = load_dict['mem']

    def sample(self, sampleSize):
        memMax = min(self.memCount, self.memSize)
        batchIndecies = np.random.choice(memMax, sampleSize, replace=False)

        states      = self.stateMemory[batchIndecies]
        actions     = self.actionMemory[batchIndecies]
        rewards     = self.rewardMemory[batchIndecies]
        nextStates  = self.nextStateMemory[batchIndecies]
        dones       = self.doneMemory[batchIndecies]

        return states, actions, rewards, nextStates, dones
