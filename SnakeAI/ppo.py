import torch 
import torch.nn as nn
import matplotlib.pyplot as plt 
import gym 
import numpy as np
from main import SnakeEnv
import warnings; warnings.filterwarnings('ignore')

torch.autograd.set_detect_anomaly(True)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, n_actions, std=0.0, discrete=False):
        super().__init__()
        self.discrete = discrete

        self.actor = nn.Sequential(
            nn.Linear(input_shape, 1024), nn.ReLU(),
            nn.Linear(1024, 1024),         nn.ReLU(),
            nn.Linear(1024, n_actions)#,   nn.Softmax(dim=-1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.to(self.device)

        self.lg_std = nn.Parameter(torch.ones(1, n_actions)*std).to(self.device)
        
        #self.apply(init_weights)

    def forward(self, state):
        state = torch.tensor([state]).float().to(self.device) if type(state) is not torch.Tensor else state
        mu = self.actor(state)
        mu = torch.softmax(mu,dim=-1)
        print(mu)
        
        if not self.discrete:
            lg_std = self.lg_std.exp().expand_as(mu)

            dist = torch.distributions.Normal(mu, lg_std)
        else:
            dist = torch.distributions.Categorical(mu)

        return dist

class CriticNetwork(nn.Module):
    def __init__(self, input_shape):
        super(CriticNetwork, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(input_shape, 1024), nn.ReLU(),
            nn.Linear(1024,1024),         nn.ReLU(),
            nn.Linear(1024, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.to(self.device)

        #self.apply(init_weights)

    def forward(self, state):
        state = torch.tensor([state]).float().to(self.device) if type(state) is not torch.Tensor else state

        return self.critic(state)


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
    def __init__(self, env):
        # SETTINGS
        self.input_shape = 800
        self.n_actions = 5

        self.env = env

        self.epochs = 4
        self.timesteps = 20
        self.epsilon = 0.2
        self.mini_batch_size = 5
        self.gamma = 0.99
        self.tau = 0.95
        self.high_score = -np.inf
        self.scores = []
        self.sc = []

        self.gae = False
        self.std_adv = False
        self.discrete = True

        self.actor = PolicyNetwork(self.input_shape, self.n_actions, discrete=self.discrete)
        self.critic = CriticNetwork(self.input_shape)
        
        self.device = self.actor.device
        
        self.memory = PPOMemory()

    def choose_action(self, state):
        dist = self.actor.forward(state)
        value = self.critic.forward(state)
        action = dist.sample()

        prob = dist.log_prob(action)

        #action = action.clamp(-1, 1)

        return action.item(), value.item(), prob.cpu().detach().numpy()


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



    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.scores)
        plt.show() 
        plt.savefig('plot.png')



    def learn(self):
        next_val = self.critic.forward(
             torch.tensor(state_).float().reshape(-1).to(agent.device)
        ).detach().cpu().numpy().tolist()[0]

        for _ in range(self.epochs):
            states, actions, rewards, values, probs, dones = self.memory.get_nps()
            
            if self.gae:
                returns =  agent.compute_gae(rewards, dones, values, next_val=next_val)
            else:
                advantages = self.compute_adv(rewards, dones, values, next_val=next_val)

            probs   =  torch.tensor(probs  ).reshape(20).detach().to(self.device)
            print(states.shape, states.dtype)
            states  =  torch.tensor(states ).float().to(self.device)
            actions =  torch.tensor(actions).reshape(20).detach().to(self.device)
            values  =  torch.tensor(values ).reshape(20).detach().to(self.device)
            dones   =  torch.tensor(dones  ).to(self.device)

            if self.gae:
                advantages = returns - values
            else:
                returns = advantages + values

            returns =  returns.reshape(20).detach().to(self.device)

            batches = self.memory.get_batches(self.mini_batch_size)

            for batch in batches:

                old_log_probs =      probs[batch]
                state         =     states[batch]
                action        =    actions[batch]
                return_       =    returns[batch]
                adv_          = advantages[batch].reshape(self.mini_batch_size, 1)
                
                if self.std_adv:
                    adv_ = (adv_ - adv_.mean()) / ( \
                                adv_.std() + 1e-4) 
    
                epsilon = 0.2
                dist   = self.actor.forward(state)
                value_ = self.critic.forward(state)
                new_log_probs = dist.log_prob(action)

                entropy = dist.entropy().mean()
                
                ratio = new_log_probs.exp() / old_log_probs.exp()
                surr1 = ratio * adv_
                surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * adv_

                a_loss = -torch.min(surr1, surr2).mean()
                c_loss = (return_ - value_)**2
                c_loss = c_loss.mean()

                total_loss = a_loss + 0.25*c_loss - 0.001*entropy
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()


env = SnakeEnv(frameRate=15)
agent = PPOAgent(env)
frame = 0
high_score = -np.inf
returns = None
for episode in range(10000):
    state = env.reset()
    done = False
    score = 0
    while not done:
        state_ = torch.tensor(state).float().to(agent.device)
        action, value, prob = agent.choose_action(state_.reshape(-1))
        print(action)
        state_, reward, done = env.step(action)
        score += reward

        agent.memory.store_memory(state.reshape(-1), action, reward, value, prob, 1-done)

        state = state_
        frame += 1
        if frame % 20 == 0:
            print('learn')
            agent.learn()
            agent.memory.reset()

    agent.sc.append(score)
    agent.scores.append(np.mean(agent.sc))
    high_score = max(high_score, score)
    avg = np.mean(agent.sc)
    print(f'episode: {episode}, high_score: {high_score}, score: {score}, avg: {avg}')

agent.plot()
