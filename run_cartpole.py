#!/usr/bin/env python
import tensorflow as tf
import gym
from DQN_Implementation import DQN_Agent
from tensorflow.python import debug as tf_debug

# Setting the session to allow growth, so it doesn't allocate all GPU memory
gpu_ops = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_ops)
sess = tf.Session(config=config)

env = gym.make('CartPole-v0')
episodes=5e3
epsilon=0.8

# Linear
# gamma = 0.99
# alpha = 0.1
# network = 'Linear' # Linear for parts 1+2, QNetwork for parts 3+4
# filepath = 'tmp/linearq/cartpole/' #'tmp/linearq_replay/cartpole/'
# replay = False # False for part 1, True for part 2

# Linear Experience Replay
# gamma = 0.99
# alpha = 0.1
# network = 'Linear' # Linear for parts 1+2, QNetwork for parts 3+4
# filepath = 'tmp/linearq_replay/cartpole/' #'tmp/linearq_replay/cartpole/'
# replay = True # False for part 1, True for part 2

# # DeepQ Network - COMPLETE
gamma = 0.99
alpha = 0.0001
network = 'DNN' # Deep network, not dueling
filepath = 'tmp/deepq/cartpole/'
replay = True 

# Initialize agent
agent = DQN_Agent(environment=env, 
                    sess=sess, 
                    network_type=network,
                    gamma=gamma,
                    filepath=filepath,
                    alpha=alpha)
agent.net.load_model_weights('model.ckpt-23245')

# Train the network
# agent.train(episodes=episodes,
#             epsilon=epsilon,
#             replay=replay)
# agent.net.save_model_weights()

agent.render=True
agent.test(episodes=20,
            epsilon=0.00)
agent.net.writer.close()


# Need to have separate epsilon decay rates for cartpole and mountain