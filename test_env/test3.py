#import gym
#env = gym.make('Enduro-v4')

import gym
import matplotlib.pyplot as plt 
from collections import deque
from gym import spaces
import numpy as np
import random 

def rgb2gray(image):
  return np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype('uint8')

class ConcatObs(gym.Wrapper):
    def __init__(self, env, k, h):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.h = h
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape # (k,h,shp[1],shp[2])
        self.observation_space = \
            spaces.Box(low=0, high=255, shape=(k,h,shp[1]), dtype=env.observation_space.dtype)


    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob[:self.h,:,:])
        return self.get_ob()

    def step(self, action):
        for _ in range(self.k):
            ob, reward, done, info = self.env.step(action)
            #self.frames.append(ob[:self.h,:,:])
            self.frames.append(ob[:self.h,:,:])
        return self.get_ob(), reward, done, info

    def get_ob(self):
        return rgb2gray(np.array(self.frames))



class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        def clip(x):
            if x < 0:
                return -1
            if x > 0:
                return 1
            return 0
        vec_clip = np.vectorize(clip)
        return vec_clip(reward)
    

height = 160
env = gym.make('ALE/Enduro-v5')
obs_space = env.observation_space
ac_space = env.action_space
print(obs_space)
print(ac_space)
obs = env.reset()[:height,:,:]
plt.imsave("1.png",obs)
random_action = env.action_space.sample()
new_obs, reward, done, info = env.step(random_action)
plt.imsave("2.png",new_obs)
print(reward, done, info)



wrapped_env = RewardWrapper(ConcatObs(env, 4, height))
# Reset the Env
obs = wrapped_env.reset()
print("Intial obs is of the shape", obs.shape)
print("The new observation space is", wrapped_env.observation_space)
plt.imsave("3.png",obs[0])
# Take one step
obs, _, _, _  = wrapped_env.step(2)
print("Obs after taking a step is", obs.shape)
plt.imsave("4.png",obs[0])

obs = wrapped_env.reset()
done = False
total_re = 0
acts = 0
while(~done):
    #action = wrapped_env.action_space.sample()
    action = 1
    obs, reward, done, info = wrapped_env.step(action)
    total_re += reward
    acts += 1
    print(f'Epoch:{acts}, Reward:{total_re}, Act:{action}')
print(total_re)