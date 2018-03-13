#!/usr/bin/env python
import tensorflow as tf
import gym
from DQN_Implementation import DQN_Agent

# Setting the session to allow growth, so it doesn't allocate all GPU memory
gpu_ops = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_ops)
sess = tf.Session(config=config)

env = gym.make('SpaceInvaders-v0')
episodes=5e4
decay_rate=9e-7

gamma = 1.0
alpha = 0.0001
epsilon = 1.0 # started with 0.5
network = 'DCNN' # Dueling Network
filepath = 'tmp/spaceinvaders/run1'
replay = True

# Initialize agent
agent = DQN_Agent(environment=env, 
                    sess=sess, 
                    network_type=network,
                    gamma=gamma,
                    filepath=filepath,
                    alpha=alpha)
agent.net.load_model_weights()

# Train the network
agent.train(episodes=episodes,
            epsilon=epsilon,
            decay_rate=decay_rate,
            replay=replay,
            check_rate=5e4,
            memory_size=500000,
            burn_in=100000)
agent.net.save_model_weights()

# agent.render=True
# agent.test(episodes=10,
#             epsilon=0.05)
agent.net.writer.close()