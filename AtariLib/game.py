import gym
import numpy as np
import torch
import math 
import time 
from utils import * 
from gymPreProcess import *
from agent import *
import matplotlib.pyplot as plt

env = make_env("PongNoFrameskip-v4", (84,84,1), 4)
agent = Agent(env.observation_space.shape, env.action_space.n)

highScore = -math.inf 
scores = []
avg_scores = []
n_games = 150
load = False
render = True

if load:
    agent.load()

for i in range(n_games):
    score = 0
    obs = env.reset()
    print(obs.shape)
    done = False 
    while not done:
        action = agent.choose_action(obs)
        obs_, reward, done, info = env.step(action)
        agent.memory.storeMemory(obs, action, reward, obs_, done)
        if render:
            env.render()
        score += reward 
        agent.train()
        obs=obs_ 

    scores.append(score)
    avg_score = np.mean(scores[-100:])
    avg_scores.append(avg_score) 
    highScore = max(highScore, score)
    print(f"Episode: {i}, Score: {score}, highScore: {highScore}, Average score: {avg_score}")

agent.save()
plotLearning([i for i in range(n_games)], avg_scores, scores, r"plots/1-150.png")
plt.show()