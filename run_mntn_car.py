#!/usr/bin/env python
import tensorflow as tf
import gym
from DQN_Implementation import DQN_Agent

# Setting the session to allow growth, so it doesn't allocate all GPU memory
gpu_ops = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_ops)
sess = tf.Session(config=config)

env = gym.make('MountainCar-v0')
episodes=1e4
epsilon=0.7
decay_rate=1e-6

# Linear only
# gamma = 1.0
# alpha = 0.01
# network = 'Linear' # Linear for parts 1+2, QNetwork for parts 3+4
# filepath = 'tmp/linearq/mountaincar/' #'tmp/linearq_replay/cartpole/'
# replay = False # False for part 1, True for part 2

# Experience Replay
# gamma = 1.0
# alpha = 0.01
# network = 'Linear' # Linear for parts 1+2, QNetwork for parts 3+4
# filepath = 'tmp/linearq_replay/mountaincar/' #'tmp/linearq_replay/cartpole/'
# replay = True # False for part 1, True for part 2

# DeepQ Network
gamma = 1.0
alpha = 0.0001
network = 'DNN' # Deep network, not dueling
filepath = 'tmp/deepq/mountaincar2/' #'tmp/linearq_replay/cartpole/'
replay = False # False for part 1, True for part 2

# Initialize agent
agent = DQN_Agent(environment=env, 
                    sess=sess, 
                    network_type=network,
                    gamma=gamma,
                    filepath=filepath)
# agent.net.load_model_weights()

# Train the network
agent.train(episodes=episodes,
            epsilon=epsilon,
            decay_rate=decay_rate,
            replay=replay)
agent.net.save_model_weights()

agent.render=True
agent.test(episodes=4,
            epsilon=0.05)
agent.net.writer.close()