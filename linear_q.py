
import gym
from DQN_Implementation import LinearQ, DQN_Agent
import numpy as np

# 

# Setup for CartPole
env = gym.make('CartPole-v0')
agent = DQN_Agent(env, 'Linear')
net = agent.net
net.gamma = 0.9
agent.epsilon = 0.5


# Setup for MountainCar
# env = gym.make('MountainCar-v0')
# agent = DQN_Agent(env, 'Linear')
# net = agent.net
# net.gamma = 1.0
# agent.epsilon = 1.0

agent.render = True

r = agent.test(episodes=2)
print(r)

for i in range(10):
    agent.train(episodes=1000)
    r = agent.test(episodes=10)    
    print(r)
env.close()