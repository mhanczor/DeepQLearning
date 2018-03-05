#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import pyglet
from gym.envs.classic_control import rendering # Have to impor this before tensorflow
import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse, time, os


class ConvQNetwork(object):

    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 

    def __init__(self, environment, alpha=0.001, gamma=1):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  
        pass
    
    def save_model_weights(self, suffix):
        # Helper function to save your model / weights. 
        pass

    def load_model(self, model_file):
        # Helper function to load an existing model.
        pass

    def load_model_weights(self,weight_file):
        # Helper funciton to load model weights. 
        pass

class QNetwork(object):

    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 

    def __init__(self, environment, alpha=0.0001, gamma=1):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  
        pass
    
    def save_model_weights(self, suffix):
        # Helper function to save your model / weights. 
        pass

    def load_model(self, model_file):
        # Helper function to load an existing model.
        pass

    def load_model_weights(self,weight_file):
        # Helper funciton to load model weights. 
        pass
        
        
class LinearQ(object):
    
    def __init__(self, environment, sess, alpha=0.0001, filepath='tmp/linearq/'):
        # Model parameters:
        self.sess = sess # the tensorflow session
        env = environment
        self.nA = env.action_space.n
        inputs = env.observation_space.shape[0]
        self.features = self.nA*inputs
        self.filepath = filepath
        
        # Set a random seed
        # tf.set_random_seed(2) # MountainCar
        
        # Linear network architecture
        with tf.name_scope("InputSpace"):
            self.x = tf.placeholder(tf.float32, [None, self.features], name='Features')
            # self.q = tf.placeholder(tf.float32, [None, 1], name='Q_Predicted')
            self.q_target = tf.placeholder(tf.float32, [None, 1], name='Q_Target')
        with tf.name_scope("Parameters"):
            self.w = tf.get_variable('Weights', shape=[self.features,1], initializer=tf.random_normal_initializer(stddev=2.0))
            self.b = tf.get_variable('Bias', shape=[1,], initializer=tf.initializers.zeros())
            # self.w = tf.Variable(np.array([[29.06],[29.04],[29.061],[-2.234],[-1.6327],[2.0925]]), dtype=tf.float32, name='Weights')
            # self.b = tf.Variable(np.array([[-92.8027]]), dtype=tf.float32, name='Bias')
        with tf.name_scope("Q_Val_Est"):
            self.q_values = tf.matmul(self.x, self.w) + self.b
        with tf.name_scope("Loss"):
            regularizer = tf.nn.l2_loss(self.w)
            self.loss = tf.losses.mean_squared_error(self.q_values, self.q_target) + 0.01*regularizer
            self.loss_sum = tf.summary.scalar("Loss", self.loss)
        with tf.name_scope("Optimize"):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.opt = tf.train.GradientDescentOptimizer(alpha).minimize(self.loss, global_step=self.global_step)  #Change this to Adam later

        self.saver = tf.train.Saver(max_to_keep=10)
        self._reset()
    
    def _reset(self):
        # Initialize the weights
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self.filepath+'events/', self.sess.graph)
        
    def infer(self, features):
        feed_dict = {self.x: features}
        pred_qval = self.sess.run(self.q_values, feed_dict=feed_dict)
        return pred_qval

    def update(self, features, q_target):
        feed_dict = {self.x: features, self.q_target: q_target}
        _, summary, w = self.sess.run([self.opt, self.loss_sum, self.w], feed_dict=feed_dict)
        return summary, w
    
    def getFeatures(self, S):
        # Create a simple feature vector that is a combination of actions and state info
        # Creates a (#actions, feature_size) matrix
        S = np.atleast_2d(S).T # Column vector
        features = np.zeros((self.nA, self.features))
        for i in range(self.nA):
            A = np.zeros((self.nA, 1)) # Action is zero-index, create one hot vector
            A[i] = 1
            features[i,:] = (np.dot(S, A.T)).reshape((-1,)) # Create a feature vector from the actions and state info
        return features
    
    def save_model_weights(self):
        self.saver.save(self.sess, self.filepath + 'checkpoints/model.ckpt', global_step=tf.train.global_step(self.sess, self.global_step))
        
    def load_model(self, model_file):
        # saver = tf.train.import_meta_graph(model_file + '.meta')
        # saver.restore(sess, model_file)
        pass
    
    def load_model_weights(self, weight_file=''):
        # Helper funciton to load model weights.
        if weight_file == '':
            filename = self.filepath+'checkpoints/'
        else:
            filename = self.filepath + 'checkpoints/' + weight_file
        
        latest_ckpt = tf.train.latest_checkpoint(filename)
        if latest_ckpt:
            self.saver.restore(self.sess, latest_ckpt)
        else:
            print('No weight file to load, starting from scratch')
            return -1
        
        self.writer = tf.summary.FileWriter('tmp/linearq', self.sess.graph)
        print('Loaded weights from {}'.format(latest_ckpt))
                

class Replay_Memory(object):

    def __init__(self, memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
        pass

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        pass

    def append(self, transition):
        # Appends transition to the memory.     
        pass

class Queue(object):
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
        
        

class DQN_Agent(object):
    
    def __init__(self, environment, sess, network_type, render=False, gamma=1., filepath='tmp/'):

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.env = environment
        self.nA = self.env.action_space.n
        self.render = render
        self.gamma = gamma
        # Initialize the memory for experience replay
                
        if network_type == 'Linear':
            self.net = LinearQ(environment, sess=sess, filepath=filepath)
        elif network_type == 'DNN':
            self.net = QNetwork(environment, sess=sess, filepath=filepath)
        elif network_type == 'DCNN':
            self.net = ConvQNetwork(environment, sess=sess, filepath=filepath)
        else:
            raise ValueError

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time. 
        return np.argmax(q_values)

    def train(self, episodes=1e3, epsilon=0.7, replay=False, check_rate=1e4):
        # Interact with the environment and update the model parameters
        # If using experience replay then update the model with a sampled minibatch
        
        if replay: # If using experience replay, need to burn in a set of transitions
            self.burn_in_memory()
            batch = 32
            
        iters = 0
        test_reward = 0
        # reward_summary = tf.Summary()
        for ep in range(int(episodes)):
            S = self.env.reset()
            done = False
            while not done:
                features = self.net.getFeatures(S)
                # Epsilon greedy training policy
                if np.random.sample() < epsilon:
                    action = np.random.choice(self.nA)
                else:
                    q_vals = self.net.infer(features)
                    action = np.argmax(q_vals)
                # Execute selected action
                S_next, R, done,_ = self.env.step(action) #DEBUG REMOVE!!!
                if not replay:
                    if done:
                        q_target = np.array([[R]])
                    else:
                        feature_next = self.net.getFeatures(S_next)
                        q_vals_next = self.net.infer(feature_next)
                        q_target = np.array([[self.gamma*np.max(q_vals_next) + R]])
                        
                    summary,_ = self.net.update(features[None,action,:], q_target) # Update the model parameters
                    self.net.writer.add_summary(summary, tf.train.global_step(self.net.sess, self.net.global_step))
                else:
                    # Update the gradient with experience replay
                    feature_cur = features[action,:]
                    # Store the tuple (feature_cur, action, R, done, feature_next)
                    # Ranomly select a batch of tuples from memory
                    # for all the tuples where done is true: set the y val as R
                    # for the remainder: set y val as R + q_max_next
                    # Update the gradients with the batch
                
                S = S_next # Update the state info
                epsilon -= 4.5e-7 # Reduce epsilon as policy learns
                
                if iters % check_rate == 0:
                    # Test the model performance
                    test_reward = self.test(episodes=20, epsilon=0.05) # Run a test to check the performance of the model
                    print('Reward: {}, Step: {}'.format(test_reward, tf.train.global_step(self.net.sess, self.net.global_step)))
                    reward_summary = tf.Summary(value=[tf.Summary.Value(tag='test_reward', simple_value=test_reward)])
                    self.net.writer.add_summary(reward_summary, tf.train.global_step(self.net.sess, self.net.global_step))
                    done = True
                iters += 1

            if ep % 100 == 0:
                print("episode {} complete, epsilon={}".format(ep, epsilon))
                print(self.net.sess.run([self.net.w, self.net.b]))
            if ep % 2000 == 0:
                self.net.save_model_weights()
        
        # cur_eps = 0
        # iters = 0
        # while cur_eps < episodes:
        #     q_values = np.zeros((self.nA, 1))
        #     for i in range(self.nA):
        #         feature_vect = self.simpleFeatures(S,i)
        #         q_values[i] = self.net.infer(feature_vect)
        # 
        #     action = self.epsilon_greedy_policy(q_values)
        #     action_features = self.simpleFeatures(S,action) #need for gradient update
        # 
        #     S, R, done,_ = self.env.step(action) # Run the simulation one step forward
        # 
        #     if done:
        #         q_target = np.array([[R]])
        #     else:
        #         # TODO Remove this duplicated effort, was copied over from a previous script
        #         for i in range(self.nA):
        #             feature_vect = self.simpleFeatures(S,i)
        #             q_values[i] = self.net.infer(feature_vect)
        #         best_act = np.argmax(q_values)
        #         q_target = np.array([self.gamma*q_values[best_act] + R])
        # 
        #     loss, weights = self.net.update(action_features, q_target) # Update the model parameters
        # 
        #     iters += 1
        #     if done:
        #         S = self.env.reset()
        #         cur_eps += 1
        #         if cur_eps % 100 == 0:
        #             print(loss)
        #             print(weights)
        #             print('{} episodes complete, {} iterations'.format(cur_eps, iters))
        # 
        #     self.epsilon -= 4.5e-6
            
            # Need to record the data for plots
            
            
        # If you are using a replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.

    def test(self, model_file=None, episodes=100, epsilon=0.0):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        total_reward = 0
        for ep in range(int(episodes)):
            episode_reward = 0
            S = self.env.reset()
            if self.render:
                self.env.render()
            done = False
            while not done:
                features = self.net.getFeatures(S)
                # Epsilon greedy training policy
                if np.random.sample() < epsilon:
                    action = np.random.choice(self.nA)
                else:
                    q_vals = self.net.infer(features)
                    action = np.argmax(q_vals)
                # Execute selected action
                S_next, R, done,_ = self.env.step(action)
                episode_reward += R
                if self.render:
                    self.env.render()
                S = S_next # Update the state
            total_reward += episode_reward
        
        average_reward = float(total_reward/episodes)
        return average_reward
    
    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        pass
        

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str, default='CartPole-v0')
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    parser.add_argument('--network',dest='network',type=str, default='Linear')
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env
    network_type = args.network

    # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)
    # sess = tf.Session()

    # You want to create an instance of the DQN_Agent class here, and then train / test it.
    env = gym.make(environment_name)
    agent = DQN_Agent(env, sess=sess, network_type=network_type)
    
    
    # agent.net.save_model_weights('test_name')
    
    # agent.render = True
    # print(sess.run(agent.net.w))
    for i in range(5):
        agent.train()
        # print(sess.run(agent.net.w))
        reward = agent.test(episodes=20)
        print("Test reward: {}".format(reward))
    env.close()
    agent.net.save_model_weights()
    

if __name__ == '__main__':
    main(sys.argv)

