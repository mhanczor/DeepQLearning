#!/usr/bin/env python
import tensorflow as tf
import gym
from DQN_Implementation import DQN_Agent
from tensorflow.python import debug as tf_debug
import numpy as np

# Setting the session to allow growth, so it doesn't allocate all GPU memory
gpu_ops = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_ops)
sess = tf.Session(config=config)

env = gym.make('CartPole-v0')
episodes=1e4
epsilon=0.8

# DeepQ Network
gamma = 0.99
alpha = 0.0001
network = 'DDNN' # Deep network, not dueling
filepath = 'tmp/cartpole/run9/'
replay = True 

# Initialize agent
agent = DQN_Agent(environment=env, 
                    sess=sess, 
                    network_type=network,
                    gamma=gamma,
                    filepath=filepath,
                    alpha=alpha,
                    double=True)
agent.net.load_model_weights()

# Train the network
agent.train(episodes=episodes,
            epsilon=epsilon,
            replay=replay,
            check_rate=5e3)
agent.net.save_model_weights()

agent.render=True
agent.test(episodes=10,
            epsilon=0.00)

# Final testing
agent.render = False
total_reward, rewards = agent.test(episodes=100, epsilon=0.00)
# print("Tested total average reward: {}".format(total_reward))
rewards = np.array(rewards)
std_dev = np.std(rewards)
mean = np.mean(rewards)
print("CartPole Mean: {}, StdDev: {}".format(mean, std_dev))

agent.net.writer.close()
agent.env.close()
sess.close()
