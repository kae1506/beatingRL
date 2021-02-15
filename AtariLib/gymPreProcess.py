import cv2 
import numpy as np
import gym
import collections

# PSEUDOCODE
'''
Class RepeatActionAndMaxFrame
  derives from: gym.Wrapper
  input: environment, repeat
  init frame buffer as an array of zeros in shape 2 x the obs space

  function step:
    input: action
  	set total reward to 0
    set done to false
  	for i in range repeat
  		call the env.step function
        receive obs, reward, done, info
  		increment total reward
  		insert obs in frame buffer
  		if done
            break
    end for
  	find the max frame
  	return: max frame, total reward, done, info

  function reset:
    input: none

  	call env.reset
  	reset the frame buffer
    store initial observation in buffer

    return: initial observation

Class PreprocessFrame
  derives from: gym.ObservationWrapper
  input: environment, new shape
  set shape by swapping channels axis
	set observation space to new shape using gym.spaces.Box (0 to 1.0)

	function observation
    input: raw observation
		covert the observation to gray scale
		resize observation to new shape
    convert observation to numpy array
    move observation's channel axis from position 2 to position 0
    observation /= 255
		return observation


Class StackFrames
  derives from: gym.ObservationWrapper
  input: environment, stack size
	init the new obs space (gym.spaces.Box) low & high bounds as repeat of n_steps
	initialize empty frame stack

	reset function
		clear the stack
		reset the environment
    for i in range(stack size)
   		append initial observation to stack
    convert stack to numpy array
    reshape stack array to observation space low shape
    return stack

	observation function
    input: observation
		append the observation to the end of the stack
		convert the stack to a numpy array
    reshape stack to observation space low shape
		return the stack of frames

function make_env:
  input: environment name, new shape, stack size
  init env with the base gym.make function
  env := RepeatActionAndMaxFrame
  env := PreprocessFrame
  env := StackFrames

  return: env
'''

'''
class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4):
        super().__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape 
        self.frame_buffer = np.zeros_like((2, self.shape))

    def step(self, action):
        t_reward = 0
        done = False 
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            idx = i%2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self):
        obs = self.env.reset()

        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = obs

        return obs

class PreprocessFrame(gym.Wrapper):
    def __init__(self, shape, env=None):
        super().__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, 
                                shape=self.shape, dtype=np.float32)

    def observation(self, obs):
        new_frame  = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs/255.0

        return new_obs


class StackFrames(gym.Wrapper):
    def __init__(self, repeat, env=None):
        super().__init__(env)
        self.repeat = repeat
        self.stack = collections.deque(maxlen=repeat)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.repeat(repeat, axis=0),
            high=env.observation_space.high.repeat(repeat, axis=0), 
            dtype=np.float32
        )

        
    def reset(self):
        self.stack.clear()
        obs = self.env.reset()

        for i in range(self.repeat):
            print(len(self.stack))
            self.stack.append(obs)

        print(len(self.stack))
        stack = np.array(self.stack).reshape(self.observation_space.low.shape)
        return stack

    def observation(self, obs):
        self.stack.append(obs)
        return np.array(self.stack).reshape(self.observation_space.low.shape)

def make_env(env, new_shape, stack_size):
    env = gym.make(env)
    env = RepeatActionAndMaxFrame(env)
    env = PreprocessFrame(new_shape, env)
    env = StackFrames(stack_size, env)

    return env
'''

import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gym

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

class RepeatActionAndMaxFrame(gym.Wrapper):
    """ modified from:
        https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/wrappers.py
    """
    def __init__(self, env=None, repeat=4, clip_reward=False,
                 no_ops=0, fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2,self.shape))
        self.clip_reward = clip_reward
        self.no_ops = 0
        self.fire_first = fire_first

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()

        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env.step(1)

        self.frame_buffer = np.zeros_like((2,self.shape))
        self.frame_buffer[0] = obs
        return obs

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape=(shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=1.0,
                                              shape=self.shape,dtype=np.float32)
    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0
        return new_obs

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                             env.observation_space.low.repeat(repeat, axis=0),
                             env.observation_space.high.repeat(repeat, axis=0),
                             dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)
        obs = np.array(self.stack).reshape(self.observation_space.low.shape)

        return obs

def make_env(env_name, shape=(44,44,1), repeat=4, clip_rewards=False,
             no_ops=0, fire_first=False):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env