#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import pyglet
import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse, time



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
    
    def __init__(self, environment, sess, alpha=0.0001, filepath='tmp/linearq'):
        # Model parameters:
        self.sess = sess # the tensorflow session
        env = environment
        self.nA = env.action_space.n
        inputs = env.observation_space.shape[0]
        self.features = self.nA*inputs
        self.filepath = filepath
        
        # Set a random seed
        tf.set_random_seed(13)
        
        # Linear network architecture
        with tf.name_scope("InputSpace"):
            self.x = tf.placeholder(tf.float32, [None, self.features], name='Features')
            # self.q = tf.placeholder(tf.float32, [None, 1], name='Q_Predicted')
            self.q_target = tf.placeholder(tf.float32, [None, 1], name='Q_Target')
        with tf.name_scope("Parameters"):
            self.w = tf.get_variable('Weights', shape=[self.features,1], initializer=tf.random_normal_initializer(stddev=2.0))
            self.b = tf.get_variable('Bias', shape=[1,], initializer=tf.initializers.zeros())
        with tf.name_scope("Q_Val_Est"):
            self.q_values = tf.matmul(self.x, self.w) + self.b
        with tf.name_scope("Loss"):
            regularizer = tf.nn.l2_loss(self.w)
            self.loss = tf.losses.mean_squared_error(self.q_values, self.q_target) + 0.01*regularizer
        with tf.name_scope("Optimize"):
            self.opt = tf.train.GradientDescentOptimizer(alpha).minimize(self.loss)  #Change this to Adam later
        
        self._reset()
    
    def _reset(self):
        # Initialize the weights
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter('tmp/linearq', self.sess.graph)
        
    def infer(self, features):
        feed_dict = {self.x: features}
        pred_qval = self.sess.run(self.q_values, feed_dict=feed_dict)
        return pred_qval

    def update(self, features, q_target):
        feed_dict = {self.x: features, self.q_target: q_target}
        _, loss, w = self.sess.run([self.opt, self.loss, self.w], feed_dict=feed_dict)
        return loss, w
    
    def _getFeatures(self, S, a):
        # Create feature vector
        # Feature vector is a combination of the inputs and selected action
        # May need to make feature vector specific to the environment?
        # Could use feature crossing from TF
        S = np.atleast_2d(S).T # Column vector
        A = np.zeros((self.nA, 1)) # Action is zero-index, create one hot vector
        A[a] = 1
        # Create a feature vector from the actions and state info
        features = np.zeros((self.features, 1))
        features = (np.dot(S, A.T)).reshape((features.shape[0],1))
    
        return features.T
    
    def save_model_weights(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.filepath + 'model.ckpt')
    
    def load_model(self, model_file):
        # saver = tf.train.import_meta_graph(model_file + '.meta')
        # saver.restore(sess, model_file)
        pass
    
    def load_model_weights(self, weight_file=''):
        # Helper funciton to load model weights. 
        if weight_file == '':
            filename = self.filepath + 'model.ckpt'
            saver.restore(self.sess, filename)
        else:
            filename = self.filepath + weight_file
            saver.restore(self.sess, filename)
        self.writer = tf.summary.FileWriter('tmp/linearq', self.sess.graph)
        print('Loaded weights from {}'.format(filename))
                

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

class DQN_Agent(object):

    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #        (a) Epsilon Greedy Policy.
    #         (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.
    
    def __init__(self, environment, sess, network_type, epsilon=0.7, gamma =1.0, render=False):

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.env = environment
        self.nA = self.env.action_space.n
        self.render = render
        self.epsilon = epsilon
        self.gamma = gamma
                
        if network_type == 'Linear':
            self.net = LinearQ(environment, sess=sess)
        elif network_type == 'DNN':
            self.net = DeepQ(environment)
        else:
            raise ValueError
        
    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        max_a = np.argmax(q_values)
        if np.random.sample() > self.epsilon:
            return max_a
        else:
            actions = [x for x in range(self.nA)]
            return np.random.choice(actions)

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time. 
        return np.argmax(q_values)

    def train(self, episodes=1e3):
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 
            
        S = self.env.reset()
        cur_eps = 0
        iters = 0
        while cur_eps < episodes:
            q_values = np.zeros((self.nA, 1))
            for i in range(self.nA):
                feature_vect = self.simpleFeatures(S,i)
                q_values[i] = self.net.infer(feature_vect)
            
            action = self.epsilon_greedy_policy(q_values)
            action_features = self.simpleFeatures(S,action) #need for gradient update

            S, R, done,_ = self.env.step(action) # Run the simulation one step forward

            if done:
                q_target = np.array([[R]])
            else:
                # TODO Remove this duplicated effort, was copied over from a previous script
                for i in range(self.nA):
                    feature_vect = self.simpleFeatures(S,i)
                    q_values[i] = self.net.infer(feature_vect)
                best_act = np.argmax(q_values)
                q_target = np.array([self.gamma*q_values[best_act] + R])

            loss, weights = self.net.update(action_features, q_target) # Update the model parameters
            
            iters += 1
            if done:
                S = self.env.reset()
                cur_eps += 1
                if cur_eps % 100 == 0:
                    print(loss)
                    print(weights)
                    print('{} episodes complete, {} iterations'.format(cur_eps, iters))
                    
            self.epsilon -= 4.5e-6
            
            # Need to record the data for plots
            
            
        # If you are using a replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.

    def test(self, model_file=None, episodes=100):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory. 
        
        cur_eps = 0
        S = self.env.reset()
        run_reward = 0
        total_reward = 0
        while cur_eps < episodes:
            q_values = np.zeros((self.nA, 1))
            for i in range(self.nA):
                # Evaluate the q values for each action at the current state
                feature_vect = self.simpleFeatures(S,i)
                q_values[i] = self.net.infer(feature_vect)
            
            action = self.greedy_policy(q_values) #Select an actions based on the epsilon greedy policy
            S, R, done,_ = self.env.step(action) # Take the next step
            run_reward += R        
            if done:
                S = self.env.reset()
                cur_eps += 1
                total_reward += run_reward 
                run_reward = 0
            if self.render:
                self.env.render()
        
        total_reward = float(total_reward/episodes)
        return total_reward

    def simpleFeatures(self, S, a):
        # Create a simple feature vector that is a combination of actions and state info
        S = np.atleast_2d(S).T # Column vector
        A = np.zeros((self.nA, 1)) # Action is zero-index, create one hot vector
        A[a] = 1
        # Create a feature vector from the actions and state info
        features = (np.dot(S, A.T)).reshape((-1,1))
        return features.T
    
    def burn_in_memory():
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str, default='MountainCar-v0')
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
    # gpu_ops = tf.GPUOptions(allow_growth=True)
    # config = tf.ConfigProto(gpu_options=gpu_ops)
    # sess = tf.Session(config=config)
    sess = tf.Session()

    # You want to create an instance of the DQN_Agent class here, and then train / test it.
    env = gym.make(environment_name)
    agent = DQN_Agent(env, sess=sess, network_type=network_type)
    
    
    # agent.net.save_model_weights('test_name')
    
    agent.render = True
    for i in range(5):
        agent.train()
        agent.test(episodes=4)
    env.close()
    agent.net.save_model_weights()
    

if __name__ == '__main__':
    main(sys.argv)

