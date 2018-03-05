from __future__ import absolute_import, division, print_function

import tensorflow as tf
import gym
# from DQN_Implementation import DQN_Agent
import numpy as np

# 

# Setup for CartPole
# env = gym.make('CartPole-v0')
# agent = DQN_Agent(env, 'Linear')
# net = agent.net
# net.gamma = 0.9
# agent.epsilon = 0.5


# Setup for MountainCar
env = gym.make('MountainCar-v0')
# agent = DQN_Agent(env, 'Linear')
# net = agent.net
# net.gamma = 1.0
# agent.epsilon = 1.0


# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
gpu_ops = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_ops)
sess = tf.Session(config=config)

x = tf.placeholder(tf.float32, [None, 2])
true = tf.placeholder(tf.float32, [None, 1])
q_val = tf.layers.dense(inputs=x, units=1, activation=None)



loss = tf.losses.mean_squared_error(true, q_val)

learning_rate = 0.001
train_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.name_scope("summaries"):
    tf.summary.scalar("loss",loss)
    merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('tmp/lr-train', tf.get_default_graph())



target = np.array([[-1]])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2):
        S = env.reset()
        done = False
        eps_R = np.zeros(1, dtype=np.float32)
        while not done:
            action = env.action_space.sample()
            S = np.atleast_2d(S)
            newS, R, done,_ = env.step(action)
            target[0] = R
            eps_R += R
            feed_dict = {x: S, true: target}
            _, summary, l = sess.run([train_opt, merged, loss], feed_dict=feed_dict)
            S = newS
        train_writer.add_summary(summary, i)
    

# agent.render = True
# 
# r = agent.test(episodes=2)
# print(r)
# 
# for i in range(10):
#     agent.train(episodes=1000)
#     r = agent.test(episodes=10)    
#     print(r)
# env.close()