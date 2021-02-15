import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt 
import time
from ppo2 import PPOAgent
from make_env import create_train_env



POS = []

def map_pos(x_pos, done, cutoff=500, clearance=10000):
    global POS
    POS.append(x_pos)

    length = len(POS)

    if length < cutoff: 
        return done

    for i in range(cutoff-1):
        if POS[length-(i+1)] != x_pos:
            return done
    
    if length > clearance:
        POS = []

    return True

env = create_train_env(1,1)

agent = PPOAgent(env, input_shape=(4, 84, 84), n_actions=5, timesteps=512, mini_batch_size=128)
frame = 0
high_score = -np.inf
returns = None
actions = np.zeros(5)

last_time = 0

STATE_SHAPE = (1,4,84,84)
N_STEPS = 200

for episode in range(100):
    state = env.reset()
    done = False
    score = 0

    while not done:
        state = state.reshape(STATE_SHAPE)
        action, value, prob = agent.choose_action(state)
        print(action)
        actions[action] += 1
        import random
        state_, reward, done, info = env.step(action)

        state_ = state_.reshape(STATE_SHAPE)
        done = map_pos(info['x_pos'], done)

        score += reward

        it = info['time'] - last_time
        last_time = info['time']

        env.render()

        agent.memory.store_memory(state, action, reward, value, prob, 1-done)

        state = state_
        frame += 1
        if frame % agent.play_steps == 0:
            agent.learn(state_)
            agent.memory.reset()

        # time.sleep(0.5)

    agent.scores.append(score)
    agent.avg_scores.append(np.mean(agent.scores))
    high_score = max(high_score, score)
    avg = np.mean(agent.scores)
    print(f'episode: {episode}, high_score: {high_score}, score: {score}, avg: {avg}, actions: {actions}')

agent.plot()
