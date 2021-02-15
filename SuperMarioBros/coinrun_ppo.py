import numpy as np
from coinrun import setup_utils, make
from ppo2 import PPOAgent
import matplotlib.pyplot as plt

def make_env(envs):
    setup_utils.setup_and_load(use_cmd_line_args=False)
    envs = make('standard', num_envs=envs)
    return envs

env = make_env(1)
#env.reset()
#print(env.observation_space.shape)
#print(env.action_space.shape)

agent = PPOAgent(env, input_shape=(3, 64, 64), n_actions=7)
scores, avgs = [], []
#agent.load_models()
high_score = -np.inf
frame = 0

STATE_SHAPE = (1, 3, 64, 64)

for i in range(500):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        obs = obs.reshape(STATE_SHAPE)
        action, value, prob = agent.choose_action(obs)

        print(action)
        obs_, reward, done, info = env.step(np.array([action]))
        
        obs_ = obs_.reshape(STATE_SHAPE)

        score += reward

        agent.memory.store_memory(obs, action, reward, value, prob, 1-done)
        env.render()
        obs = obs_

        frame += 1

        if frame % agent.play_steps == 0:
            print('learn')
            agent.learn(obs_)
            agent.memory.reset()

    scores.append(score)
    avg = np.mean(scores)
    avgs.append(avg)
    high_score = max(high_score, score)
    print(f'episode: {i}, high_score: {high_score}, score: {score}, avg: {avg}')

    if i % 20 == 0 and i != 0:
        agent.save_models()

plt.cla()
plt.plot(avgs)
plt.show()
agent.save_models()
