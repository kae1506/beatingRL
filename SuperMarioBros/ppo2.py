import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gym
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from pyutils.torch_utils import ThreeByThree

torch.autograd.set_detect_anomaly(True)

class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self.n_actions = n_actions

        #self.actor = nn.Sequential(
        #    nn.Linear(input_shape, 256), nn.Tanh(),
        #    nn.Linear(256, 256),         nn.Tanh(),
        #    nn.Linear(256, n_actions)#,  nn.Softmax(dim=-1)
        #)

        self.device = torch.device('cpu' if torch.cuda.is_available else 'cpu')

        self.actor = ThreeByThree(input_shape, n_actions, device=self.device, fc1=512, fc2=512)
  
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.to(self.device)

    def forward(self, state, noise_scale=1.0):
        state = torch.tensor(state).float().to(self.device) if type(state) is not torch.Tensor else state
        mu = self.actor(state).squeeze(0)
        
        noise = torch.tensor(
            torch.distributions.Normal(mu, noise_scale).sample()
        ).to(self.device)

        mu += noise

        mu = torch.softmax(mu, dim=-1)


        dist = torch.distributions.Categorical(mu)

        return dist

class CriticNetwork(nn.Module):
    def __init__(self, input_shape):
        super(CriticNetwork, self).__init__()
        
        #self.critic = nn.Sequential(
        #    nn.Linear(input_shape, 256), nn.ReLU(),
        #    nn.Linear(256,256),         nn.ReLU(),
        #    nn.Linear(256, 1)
        #)

        self.device = torch.device('cpu' if torch.cuda.is_available else 'cpu')

        self.critic = ThreeByThree(input_shape, 1, device=self.device, fc1=512, fc2=512)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.to(self.device)

    def forward(self, state):
        state = torch.tensor(state).float().to(self.device) if type(state) is not torch.Tensor else state

        return self.critic(state).squeeze(0)


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

        self.value = []
        self.probs = []
        self.dones = []

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []

        self.value = []
        self.probs = []
        self.dones = []

    def store_memory(self, s, a, r, v, p, d):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

        self.value.append(v)
        self.probs.append(p)
        self.dones.append(d)

    def get_batches(self, batch_size):
        t_l = len(self.dones)
        indices = np.arange(t_l, dtype=np.float32)
        np.random.shuffle(indices)
        start_indicies = np.arange(0, t_l, batch_size)
        batches = [indices[i:i+batch_size] for i in start_indicies]

        return batches

    def get_nps(self):
        return np.array(self.states), \
                np.array(self.actions), \
                np.array(self.rewards), \
                np.array(self.value), \
                np.array(self.probs), \
                np.array(self.dones) 

class PPOAgent:
    def __init__(self,
            env,
            input_shape=None,
            n_actions=None,
            epochs=4,
            timesteps=512,
            mini_batch_size=128,
            gamma=0.99,
            tau=0.95,
            noise_scale=1.0):

        # SETTINGS
        self.input_shape = input_shape if input_shape else 4
        self.n_actions = n_actions if n_actions else 2
    
        self.filename = 'models/'
        self.env = env

        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = 1.0
        self.play_steps = timesteps
        
        self.adv_norm = False
        self.gae = False
        
        self.high_score = -np.inf
        self.avg_scores = []
        self.scores = []

        self.actor = PolicyNetwork(self.input_shape, self.n_actions)
        self.critic = CriticNetwork(self.input_shape)
        
        self.device = self.actor.device
        
        self.memory = PPOMemory()

    def save_models(self):
        print('__saving__')
        torch.save(self.actor.state_dict(), self.filename+'actor.h')
        torch.save(self.critic.state_dict(), self.filename+'critic.h')

    def load_models(self):
        print('__loading__')
        self.actor.load_state_dict(torch.load(self.filename+'actor.h'))
        self.critic.load_state_dict(torch.load(self.filename+'critic.h'))

    def choose_action(self, state):
        dist = self.actor.forward(state, noise_scale=self.noise_scale)
        #print(dist)
        value = self.critic.forward(state)
        action = dist.sample()
        
        prob = dist.log_prob(action)
        #print(prob.shape)
#
        self.noise_scale -= 0.0000001 if self.noise_scale > 0.001 else 0.001

        action = action.detach().cpu().numpy()
        prob = prob.detach().cpu().numpy()
        return int(action), value.detach().cpu().numpy(), prob

    def compute_gae(self, rewards, masks, values, next_val=None):
        returns = []
        gae = 0
        value_ = np.append(values, next_val)

        for i in reversed(range(len(rewards))):
            td_res = rewards[i] + self.gamma * value_[i+1] * masks[i] - value_[i]
            gae = td_res + self.gamma * self.tau * masks[i] * gae
            returns.insert(0, gae+value_[i])

        return torch.tensor(returns).to(self.device)
        
    def compute_adv(self, rewards, masks, values, next_val=None):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0
        value_ = np.append(values, next_val)

        for i in reversed(range(len(rewards))):
           td_res = rewards[i] + self.gamma * value_[i+1] * masks[i] - value_[i]
           gae = td_res + self.gamma * self.tau * masks[i] * gae
           advantages[i] = gae

        return torch.tensor(advantages).to(self.device)

    def plot(self, filename="SuperMarioBros-run0.png"):
        fig = plt.figure()
        ax = fig.add_subplot(111, label='1')
        ax2 = fig.add_subplot(111, label='2', frame_on=False)

        ax.plot(self.scores)
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Scores')
        ax2.plot(self.avg_scores)
        ax2.set_ylabel('Avg Scores')

    def learn(self, state_):
        next_val = self.critic.forward(
             torch.tensor(state_).float().to(self.device)
        ).detach().cpu().numpy().tolist()[0]

        for _ in range(self.epochs):
            states, actions, rewards, values, probs, dones = self.memory.get_nps()
            
            if self.gae:
                returns =  agent.compute_gae(rewards, dones, values, next_val=next_val)
            else:
                advantages = self.compute_adv(rewards, dones, values, next_val=next_val)

            probs   =  torch.tensor(probs  ).reshape(self.play_steps).detach().to(self.device)
            states  =  torch.tensor(states ).float().to(self.device)
            actions =  torch.tensor(actions).reshape(self.play_steps).detach().to(self.device)
            values  =  torch.tensor(values ).reshape(self.play_steps).detach().to(self.device)
            dones   =  torch.tensor(dones  ).to(self.device)

            if self.gae:
                advantages = returns - values
            else: 
                returns = advantages + values

            returns =  returns.reshape(self.play_steps).detach().to(self.device)

            batches = self.memory.get_batches(self.mini_batch_size)

            for batch in batches:
                input_shape = self.input_shape
                old_log_probs =      probs[batch]
                state         =     states[batch].reshape(self.mini_batch_size, input_shape[0], input_shape[1], input_shape[2])
                action        =    actions[batch]
                return_       =    returns[batch]
                adv_          = advantages[batch].reshape(self.mini_batch_size, 1)
                
                if self.adv_norm:
                    adv_ = (adv_ - adv_.mean()) / ( \
                                     adv_.std() + 1e-4)

                epsilon = 0.2
                dist   = self.actor.forward(state, noise_scale=self.noise_scale)
                value_ = self.critic.forward(state)
                new_log_probs = dist.log_prob(action)

                entropy = dist.entropy().mean()
                
                ratio = new_log_probs.exp() / old_log_probs.exp()
                surr1 = ratio * adv_
                surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * adv_

                a_loss = -torch.min(surr1, surr2).mean()
                c_loss = (return_ - value_)**2
                c_loss = c_loss.mean()

                total_loss = a_loss + 0.5*c_loss - 0.001*entropy
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward(retain_graph=False)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

if __name__ == '__main__':
    def animate(i):
        plt.cla()
        plt.plot(agent.avg_scores)

    env = gym.make('CartPole-v1').unwrapped
    agent = PPOAgent(env)
    frame = 0
    high_score = -np.inf
    returns = None

    ani = FuncAnimation(plt.gcf(), animate, interval=1000)


    for episode in range(1000):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action, value, prob = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            score += reward

            agent.memory.store_memory(state, action, reward, value, prob, 1-done)
            #env.render()
            state = state_
            frame += 1
            if frame % agent.play_steps == 0:
                print('learn')
                agent.learn(state_)
                agent.memory.reset()

        agent.scores.append(score)
        agent.avg_scores.append(np.mean(agent.scores))
        high_score = max(high_score, score)
        avg = np.mean(agent.scores)

        print(f'episode: {episode}, high_score: {high_score}, score: {score}, avg: {avg}')

    plt.show()
