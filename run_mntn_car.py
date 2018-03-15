#!/usr/bin/env python
import tensorflow as tf
import gym
from DQN_Implementation import DQN_Agent
import numpy as np

# Setting the session to allow growth, so it doesn't allocate all GPU memory
gpu_ops = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_ops)
sess = tf.Session(config=config)

env = gym.make('MountainCar-v0')
episodes=5e3
decay_rate=1e-6

# Dueling DeepQ Network
gamma = 1.0
alpha = 0.00001 # started with 0.0001
epsilon = 0.3 # started with 0.5
network = 'DDNN' # Dueling Network
filepath = 'tmp/dueling-q/mountaincar/' #'tmp/linearq_replay/cartpole/'
replay = True

# Initialize agent
agent = DQN_Agent(environment=env, 
                    sess=sess, 
                    network_type=network,
                    gamma=gamma,
                    filepath=filepath,
                    alpha=alpha)
agent.net.load_model_weights('model.ckpt-200001')

# Train the network
agent.train(episodes=episodes,
            epsilon=epsilon,
            decay_rate=decay_rate,
            replay=replay)
agent.net.save_model_weights()

agent.render=True
agent.test(episodes=20,
            epsilon=0.00)
            
# Final testing
agent.render = False
total_reward, rewards = agent.test(episodes=100, epsilon=0.05)
rewards = np.array(rewards)
std_dev = np.std(rewards)
mean = np.mean(rewards)
print("MountainCar Mean: {}, StdDev: {}".format(mean, std_dev))

agent.net.writer.close()
agent.env.close()
sess.close()