#!/usr/bin/env python
import tensorflow as tf
import gym
from DQN_Implementation import DQN_Agent
import numpy as np

# Setting the session to allow growth, so it doesn't allocate all GPU memory
gpu_ops = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=True, gpu_options=gpu_ops)
config = tf.ConfigProto(gpu_options=gpu_ops)
sess = tf.Session(config=config)

# env = gym.make('SpaceInvaders-v0')
env = gym.make('SpaceInvadersDeterministic-v0')
episodes = 5e3
decay_rate = 9e-7

gamma = 0.99
alpha = 0.00025
epsilon = 1.0 # started with 0.5
network = 'DCNN' # Dueling Network
filepath = 'tmp/spaceinvaders/run1/'
replay = True
double = True

# Initialize agent
agent = DQN_Agent(environment=env, 
                    sess=sess, 
                    network_type=network,
                    gamma=gamma,
                    filepath=filepath,
                    alpha=alpha,
                    double=double)
# agent.net.load_model_weights()

# Train the network
agent.train(episodes=episodes,
            epsilon=epsilon,
            decay_rate=decay_rate,
            replay=replay,
            check_rate=1e4,
            memory_size=500000,
            burn_in=50000)
agent.net.save_model_weights()

# agent.render=True
# agent.test(episodes=10,
#             epsilon=0.05)

# Final testing
agent.render = False
total_reward, rewards = agent.test(episodes=100, epsilon=0.05)
rewards = np.array(rewards)
std_dev = np.std(rewards)
mean = np.mean(rewards)
print("Atari Mean: {}, StdDev: {}".format(mean, std_dev))

agent.net.writer.close()
agent.env.close()
sess.close()