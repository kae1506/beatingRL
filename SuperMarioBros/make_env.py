from pyutils.stack_gym import StackGym
from pyutils.gym_utils import Monitor

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT

from nes_py.wrappers import JoypadSpace

import torch.multiprocessing as mp
import subprocess as sp 

import numpy as np
import cv2

from gym.spaces import Box
from gym import Wrapper


import matplotlib.pyplot as plt


class MultipleEnvironments:
    def __init__(self, world, stage, action_type, num_envs, output_path=None):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        if action_type == "right":
            actions = RIGHT_ONLY
        elif action_type == "simple":
            actions = SIMPLE_MOVEMENT
        else:
            actions = COMPLEX_MOVEMENT
        self.envs = [create_train_env(world, stage, actions, output_path=output_path) for _ in range(num_envs)]
        self.num_states = self.envs[0].observation_space.shape[0]
        self.num_actions = len(actions)
        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index,))
            process.start()
            self.env_conns[index].close()

    def run(self, index):
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(self.envs[index].step(action.item()))
            elif request == "reset":
                self.env_conns[index].send(self.envs[index].reset())
            else:
                raise NotImplementedError



class CustomReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)
        state = process_frame(state)
        #reward += (info["score"] - self.curr_score) / 20
        #reward = np.clip(reward, -3, 3)
        # self.curr_score = info["score"]
        # if done:
        #     if info["flag_get"]:
        #         reward += 50
        #     else:
        #         reward -= 50
        return state, reward, done, info

        #return state, reward, done, info

    def reset(self):
        self.curr_score = 0
        return process_frame(self.env.reset())

def process_frame(frame):
    if frame is not None:
        # plt.imshow(frame.reshape(84, 84, 1) if frame.shape == (1,84,84) else frame)
        # plt.show()
        frame = frame.reshape(240, 256, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))

def create_train_env(world, stage, actions=RIGHT_ONLY, output_path=None):
    print(actions)
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None

    env = JoypadSpace(env, actions)
    #print(env.action_space.sample())
    env = CustomReward(env, monitor)
    env = StackGym(env, frame_stack_size=4, state_formatter=None)
    return env