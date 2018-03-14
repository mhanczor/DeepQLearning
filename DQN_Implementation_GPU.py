#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import pyglet
from gym.envs.classic_control import rendering # Have to impor this before tensorflow
import tensorflow as tf
import numpy as np
import gym, sys, copy, argparse, time, os
import random
from PIL import Image


class ConvQNetwork(object):
    
    # DeepQ Network for solving games with visual input like SpaceInvaders
    def __init__(self, environment, sess, alpha=0.001, filepath='tmp/convq/', is_dueling=True):
        self.sess = sess
        env = environment
        self.nA = env.action_space.n
        self.nObs = (None,) + env.observation_space.shape
        self.filepath = filepath
        
        channels = 4
        self.image_size = [84, 84] # The image size to scale down to
        self.frame_buffer = np.zeros(([1] + self.image_size + [channels]), dtype=np.uint8)
        
        with tf.name_scope("Input"):
            input_size = [None] + self.image_size + [channels]
            self.x = tf.placeholder(tf.float32, input_size, name='Features')
            self.q_target = tf.placeholder(tf.float32, [None, 1], name='Q_Target')
            self.action = tf.placeholder(tf.int32, [None], name='Selected_Action')
        with tf.name_scope("Conv_Layers"):
            # 84x84x4 input
            cnv_1 = tf.layers.conv2d(self.x, filters=32, kernel_size=[8,8], strides=[4, 4])  #20x20x32
            cnv_2 = tf.layers.conv2d(cnv_1, filters=64, kernel_size=[4,4], strides=[2, 2]) #9x9x64
            cnv_3 = tf.layers.conv2d(cnv_2, filters=64, kernel_size=[3,3], strides=[1, 1]) #7x7x64
            cnv_3_flat = tf.reshape(cnv_3, [-1, 7 * 7 * 64])
        with tf.name_scope("Layers"):
            fc_1 = tf.layers.dense(inputs=cnv_3_flat, units=512, activation=tf.nn.relu)
        with tf.name_scope("Output"):
            if is_dueling:
                # Dueling DQN
                fc_a1 = tf.layers.dense(inputs=fc_1, units=512, activation=tf.nn.relu)
                advantage = tf.layers.dense(inputs=fc_a1, units=self.nA)
                self.advantage = tf.subtract(advantage, tf.reduce_mean(advantage))
                fc_v1 = tf.layers.dense(inputs=fc_1, units=512, activation=tf.nn.relu)
                value = tf.layers.dense(inputs=fc_v1, units=1)
                self.q_pred = tf.add(value, self.advantage)
                self.q_onehot = tf.one_hot(self.action, self.nA, axis=-1)
                self.q_action = tf.reduce_sum(tf.multiply(self.q_onehot, self.q_pred), 1, keepdims=True)
            else:
                # Vanilla DQN
                self.q_pred = tf.layers.dense(inputs=fc_1, units=self.nA)
                self.q_onehot = tf.one_hot(self.action, self.nA, axis=-1)
                self.q_action = tf.reduce_sum(tf.multiply(self.q_onehot, self.q_pred), 1, keepdims=True) # Qval for the chosen action, should have a dimension (None, 1)
            with tf.name_scope("Loss"):
                self.loss = tf.losses.mean_squared_error(self.q_target, self.q_action)
                self.loss_summary = tf.summary.scalar("Loss", self.loss)
            with tf.name_scope("Optimize"):
                # Clip gradient norm to be leq 10
                train_opt = tf.train.AdamOptimizer(alpha)
                grads = train_opt.compute_gradients(self.loss)
                capped_grads = [(tf.clip_by_norm(grad, 10), var) for grad, var in grads]
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                self.opt = train_opt.apply_gradients(capped_grads, global_step=self.global_step)
                
                # self.opt = tf.train.AdamOptimizer(alpha).minimize(self.loss, global_step=self.global_step)
            
            self.saver = tf.train.Saver(max_to_keep=20)
            self._reset()
        
    def _reset(self):
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self.filepath+'events/', self.sess.graph)
    
    def infer(self, features):
        # Evaluate the data using the model
        feed_dict = {self.x: features} # Features is a (batch, obs_space) matrix
        q_vals = self.sess.run(self.q_pred, feed_dict=feed_dict)
        return q_vals
    
    def update(self, features, q_target, action=None):
        # Update the model by calculating the loss over a selected action
        action = action.flatten() # Actions must be in a 1d aray
        feed_dict = {self.x: features, self.action: action, self.q_target: q_target}
        _, loss_summary, loss, onehot, act, pred, target, q_act = self.sess.run([self.opt, self.loss_summary, self.loss, self.q_onehot, self.action, self.q_pred, self.q_target, self.q_action], feed_dict=feed_dict)
        # _, loss_summary, loss, onehot, act, pred, target, q_act, new_tar  = self.sess.run([self.opt, self.loss_summary, self.loss, self.q_onehot, self.action, self.q_pred, self.q_target, self.q_action, self.new_target], feed_dict=feed_dict)
        # print("Predicted: {}, Onehot: {}, Action: {}, Q-Act: {}, Target: {}, Loss: {}".format(pred, onehot, act, q_act, new_tar, loss))
        return loss_summary, loss
    
    def getFeatures(self, S):
        # Need to convert raw image input into 4 stacked frames
        im = Image.fromarray(S)
        im = im.convert(mode='L')
        box = [15, 10, im.width-15, im.height-12] # From experiments with the images
        im = im.crop(box=box)
        im = im.resize([84,84], resample=2)
        reduced_frame = np.asarray(im, dtype=np.uint8)

        # Should append to the frame queue
        self.frame_buffer = np.roll(self.frame_buffer, 1, 3)
        self.frame_buffer[0,:,:,0] = reduced_frame
        
        #Should only be processing the most recent frame, use it like a queue
        return self.frame_buffer
    
    def save_model_weights(self):
        # Helper function to save your model / weights. 
        self.saver.save(self.sess, self.filepath + 'checkpoints/model.ckpt', global_step=tf.train.global_step(self.sess, self.global_step))
        print("Saved Weights")
        
    def load_model(self, model_file=''):
        # Helper function to load an existing model.
        
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
            print('Loaded weights from {}'.format(latest_ckpt))
        elif weight_file != '':
            try:
                self.saver.restore(self.sess, filename)
                print('Loaded weights from {}'.format(filename))
            except:
                print("Loading didn't work")
        else:
            print('No weight file to load, starting from scratch')
            return -1
        
        self.writer = tf.summary.FileWriter(self.filepath+'events/', self.sess.graph)

class QNetwork(object):
    
    # Deep Q network for solving environments MountainCar and CartPole
    # Take in state information as an input, output q-value for each action

    def __init__(self, environment, sess, alpha=0.0001, filepath='tmp/deepq/', is_dueling=False):
        
        self.sess = sess
        env = environment
        self.nA = env.action_space.n
        self.nObs = (None,) + env.observation_space.shape
        self.filepath = filepath
        
        # tf.set_random_seed(9) # good for 30s
        
        with tf.name_scope("Input"):
            self.x = tf.placeholder(tf.float32, self.nObs, name='Features')
            self.q_target = tf.placeholder(tf.float32, [None, 1], name='Q_Target')
            self.action = tf.placeholder(tf.int32, [None], name='Selected_Action')
        with tf.name_scope("Layers"):
            fc_1 = tf.layers.dense(inputs=self.x, units=30, activation=tf.nn.relu)
            fc_2 = tf.layers.dense(inputs=fc_1, units=30, activation=tf.nn.relu)
            fc_3 = tf.layers.dense(inputs=fc_2, units=30, activation=tf.nn.relu)
        with tf.name_scope("Output"):
            
            if is_dueling:
                # Dueling DQN
                advantage = tf.layers.dense(inputs=fc_3, units=self.nA)
                self.advantage = tf.subtract(advantage, tf.reduce_mean(advantage))
                value = tf.layers.dense(inputs=fc_3, units=1)
                self.q_pred = tf.add(value, self.advantage)
                self.q_onehot = tf.one_hot(self.action, self.nA, axis=-1)
                self.q_action = tf.reduce_sum(tf.multiply(self.q_onehot, self.q_pred), 1, keepdims=True)
            else:
                # Vanilla DQN
                self.q_pred = tf.layers.dense(inputs=fc_3, units=self.nA)
                self.q_onehot = tf.one_hot(self.action, self.nA, axis=-1)
                self.q_action = tf.reduce_sum(tf.multiply(self.q_onehot, self.q_pred), 1, keepdims=True) # Qval for the chosen action, should have a dimension (None, 1)
                
        with tf.name_scope("Loss"):
            regularizer = 0.1*(tf.nn.l2_loss(fc_1) + tf.nn.l2_loss(fc_2) + tf.nn.l2_loss(fc_3))
            self.loss = tf.losses.mean_squared_error(self.q_target, self.q_action)
            # self.loss = tf.reduce_mean(tf.losses.huber_loss(self.q_target, self.q_action, delta=2000))
            self.loss_summary = tf.summary.scalar("Loss", self.loss)
        with tf.name_scope("Optimize"):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.opt = tf.train.AdamOptimizer(alpha).minimize(self.loss, global_step=self.global_step)
        
        self.saver = tf.train.Saver(max_to_keep=10)
        self._reset()
    
    def _reset(self):
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self.filepath+'events/', self.sess.graph)
        
    def infer(self, features):
        # Evaluate the data using the model
        feed_dict = {self.x: features} # Features is a (batch, obs_space) matrix
        q_vals = self.sess.run(self.q_pred, feed_dict=feed_dict)
        return q_vals
        
    def update(self, features, q_target, action=None):
        # Update the model by calculating the loss over a selected action
        action = action.flatten() # Actions must be in a 1d aray
        feed_dict = {self.x: features, self.action: action, self.q_target: q_target}
        _, loss_summary, loss, onehot, act, pred, target, q_act = self.sess.run([self.opt, self.loss_summary, self.loss, self.q_onehot, self.action, self.q_pred, self.q_target, self.q_action], feed_dict=feed_dict)
        # _, loss_summary, loss, onehot, act, pred, target, q_act, new_tar  = self.sess.run([self.opt, self.loss_summary, self.loss, self.q_onehot, self.action, self.q_pred, self.q_target, self.q_action, self.new_target], feed_dict=feed_dict)
        # print("Predicted: {}, Onehot: {}, Action: {}, Q-Act: {}, Target: {}, Loss: {}".format(pred, onehot, act, q_act, new_tar, loss))
        return loss_summary, loss
        
    def getFeatures(self, S):
        # Used here to make agent compatible with multiple state information types
        return np.atleast_2d(S)
    
    def save_model_weights(self):
        # Helper function to save your model / weights. 
        self.saver.save(self.sess, self.filepath + 'checkpoints/model.ckpt', global_step=tf.train.global_step(self.sess, self.global_step))
        print("Saved Weights")
        
    def load_model(self, model_file):
        # Helper function to load an existing model.
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
            print('Loaded weights from {}'.format(latest_ckpt))
        elif weight_file != '':
            try:
                self.saver.restore(self.sess, filename)
                print('Loaded weights from {}'.format(filename))
            except:
                print("Loading didn't work")
        else:
            print('No weight file to load, starting from scratch')
            return -1
        
        self.writer = tf.summary.FileWriter(self.filepath+'events/', self.sess.graph)
        
        
        
class LinearQ(object):
    
    def __init__(self, environment, sess, alpha=0.0001, filepath='tmp/linearq/'):
        # Model parameters:
        self.sess = sess
        env = environment
        self.nA = env.action_space.n
        inputs = env.observation_space.shape[0]
        self.features = self.nA*inputs
        self.filepath = filepath
        
        # Set a random seed
        tf.set_random_seed(2) # MountainCar No Replay
        # tf.set_random_seed(7)
        
        # Linear network architecture
        with tf.name_scope("InputSpace"):
            self.x = tf.placeholder(tf.float32, [None, self.features], name='Features')
            self.q_target = tf.placeholder(tf.float32, [None, 1], name='Q_Target')
        with tf.name_scope("Parameters"):
            self.w = tf.get_variable('Weights', shape=[self.features,1], initializer=tf.random_normal_initializer(stddev=2.0))
            self.b = tf.get_variable('Bias', shape=[1,], initializer=tf.initializers.zeros())
        with tf.name_scope("Q_Val_Est"):
            self.q_values = tf.matmul(self.x, self.w) + self.b
        with tf.name_scope("Loss"):
            regularizer = tf.nn.l2_loss(self.w)
            self.loss = tf.losses.mean_squared_error(self.q_values, self.q_target) + 0.01*regularizer
            self.loss_sum = tf.summary.scalar("Loss", self.loss)
        with tf.name_scope("Optimize"):
            
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            # self.opt = tf.train.GradientDescentOptimizer(alpha).minimize(self.loss, global_step=self.global_step)  #Change this to Adam later
            train_opt = tf.train.AdamOptimizer(alpha)
            grads = train_opt.compute_gradients(self.loss)
            capped_grads = [(tf.clip_by_value(grad, -100., 100.), var) for grad, var in grads]
            self.opt = train_opt.apply_gradients(capped_grads, global_step=self.global_step)
            
            # self.opt = tf.train.AdamOptimizer(alpha).minimize(self.loss, global_step=self.global_step)

        self.saver = tf.train.Saver(max_to_keep=20)
        self._reset()
    
    def _reset(self):
        # Initialize the weights
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(self.filepath+'events/', self.sess.graph)
        
    def infer(self, features):
        feed_dict = {self.x: features}
        pred_qval = self.sess.run(self.q_values, feed_dict=feed_dict)
        return pred_qval

    def update(self, features, q_target, action=None):
        feed_dict = {self.x: features, self.q_target: q_target}
        _, summary, loss, w = self.sess.run([self.opt, self.loss_sum, self.loss, self.w], feed_dict=feed_dict)
        # print(w)
        # raw_input()
        return summary, loss
    
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
            print('Loaded weights from {}'.format(latest_ckpt))
        elif weight_file != '':
            try:
                self.saver.restore(self.sess, filename)
                print('Loaded weights from {}'.format(filename))
            except:
                print("Loading didn't work")
        else:
            print('No weight file to load, starting from scratch')
            return -1
        
        self.writer = tf.summary.FileWriter(self.filepath+'events/', self.sess.graph)
                

class Replay_Memory(object):

    def __init__(self, memory_size=50000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.
        self.memory = []
        self.memory_size = memory_size
        self.feature_shape = None

    def sample_batch(self, batch_size=32, is_linear=True):    
        # Return the data in matrix forms for easy feeding into networks
        # Data should be in in form (batch, values)
        if batch_size < 1 or batch_size >= self.size():
            raise ValueError
        
        batch = random.sample(self.memory, batch_size)
        batch_features = (batch_size,) + self.feature_shape[1:]
        cur_features = np.zeros(batch_features, dtype=np.uint8)
        actions = np.zeros((batch_size, 1), dtype=np.uint8)
        rewards = np.zeros((batch_size, 1), dtype=np.uint8)
        dones = np.zeros((batch_size, 1), dtype=np.bool)
        if is_linear:
            next_features = []
        else:
            next_features = np.zeros(batch_features, dtype=np.uint8)
        
        for i, ele in enumerate(batch):
            cur_features[i,:] = ele[0]
            actions[i,:] = ele[1]
            rewards[i,:] = ele[2]
            dones[i,:] = ele[3]
            if is_linear:
                next_features += (ele[4],) # For linear state and action features
            else:
                next_features[i,:] = ele[4]   
        return (cur_features, actions, rewards, dones, next_features)

    def append(self, transition):
        
        if not self.feature_shape: #On the first entry record the shape of the features
            self.feature_shape = transition[0].shape
                    
        # Appends transition to the memory.
        if self.size() >= self.memory_size:
              self.memory.pop()
        self.memory.insert(0,transition)
        
        if self.size() > self.memory_size:
            print('Queue Overfilled')
        
    def size(self):
        return len(self.memory)
                

class DQN_Agent(object):
    
    def __init__(self, environment, sess, network_type, render=False, gamma=1., alpha=0.001, filepath='tmp/'):

        self.env = environment
        self.nA = self.env.action_space.n
        self.render = render
        self.gamma = gamma
                
        if network_type == 'Linear':
            self.net = LinearQ(environment, sess=sess, filepath=filepath, alpha=alpha)
            self.linear = True
        elif network_type == 'DNN':
            self.net = QNetwork(environment, sess=sess, filepath=filepath, alpha=alpha)
            self.linear = False
        elif network_type == 'DDNN':
            self.net = QNetwork(environment, sess=sess, filepath=filepath, alpha=alpha, is_dueling=True)
            self.linear = False
        elif network_type == 'DCNN':
            self.net = ConvQNetwork(environment, sess=sess, filepath=filepath, alpha=alpha)
            self.linear = False
        else:
            raise ValueError

    def train(self, episodes=1e3, epsilon=0.7, decay_rate=4.5e-6, replay=False, check_rate=1e4, memory_size=50000, burn_in=10000):
        # Interact with the environment and update the model parameters
        # If using experience replay then update the model with a sampled minibatch
        
        if replay: # If using experience replay, need to burn in a set of transitions
            if self.linear:
                memory_queue = Replay_Memory()
                self.burn_in_memory(memory_queue, burn_in=100)
                batch_size = 32
                print('Memory Burned In')
            else:
                memory_queue = Replay_Memory(memory_size=memory_size)
                self.burn_in_memory(memory_queue, burn_in=burn_in)
                batch_size = 32
                print('Memory Burned In')
            
        iters = 0
        test_reward = 0
        reward_summary = tf.Summary()
        ep_reward_summary = tf.Summary()
        for ep in range(int(episodes)):
            ep_reward = 0
            S = self.env.reset()
            if not self.linear:
                for i in range(4):
                    features = self.net.getFeatures(S) # Fill the Atari buffer with frames
            done = False
            while not done:
                features = self.net.getFeatures(S)
                # Epsilon greedy training policy
                if np.random.sample() < epsilon:
                    action = np.random.randint(self.nA)
                else:
                    q_vals = self.net.infer(features)
                    action = np.argmax(q_vals)
                # Execute selected action
                S_next, R, done,_ = self.env.step(action)
                #Normalize rewards for SpaceInvaders
                # if R > 0:
                #     R = 1
                # elif R < 0:
                #     R = -1
                # else:
                #     R = 0
                ep_reward += R
                if not replay:
                    if done:
                        q_target = np.array([[R]])
                    else:
                        feature_next = self.net.getFeatures(S_next)
                        q_vals_next = self.net.infer(feature_next)
                        q_target = np.array([[self.gamma*np.max(q_vals_next) + R]])
                        
                    if self.linear:
                        features = features[None,action,:]
                    summary, loss = self.net.update(features, q_target, action=np.array([[action]])) 
                    if np.isnan(loss):
                        print("Loss exploded")
                        return          
                    
                    self.net.writer.add_summary(summary, tf.train.global_step(self.net.sess, self.net.global_step))
                else:
                    # Update the gradient with experience replay
                    if self.linear:
                        features = features[None,action,:]
                    feature_next = self.net.getFeatures(S_next)
                    store = (features, action, R, done, feature_next)
                    # Store the tuple (features, action, R, done, feature_next)
                    memory_queue.append(store)
                    
                    # Ranomly select a batch of tuples from memory
                    cur_features, actions, rewards, dones, next_features = memory_queue.sample_batch(batch_size=batch_size, is_linear=self.linear)
                    
                    # Need to differentiate between features for linear and deep models
                    # Features for the linear model are state and action dependent
                    if self.linear:
                        best_q = np.zeros((batch_size,1))
                        for i, ele in enumerate(next_features):
                            best_q[i,:] = np.max(self.net.infer(ele))
                    else:
                        best_q = self.net.infer(next_features)
                        best_q = np.max(best_q, axis=1, keepdims=True)
                    
                    # For atari:
                    # Atari the evaluate is going to take in one feature and spit out a list of q_vals
                                        
                    done_mask = 1 - dones.astype(int) # Makes a mask of 0 where done is true, 1 otherwise
                    q_target = self.gamma*best_q * done_mask + rewards # If done, target just reward, else target reward + best_q
                    
                    # Update the gradients with the batch
                    summary, loss = self.net.update(cur_features, q_target, action=actions)
                    if np.isnan(loss):
                        print("Loss exploded")
                        return
                    self.net.writer.add_summary(summary, tf.train.global_step(self.net.sess, self.net.global_step))
                    # print(loss)
                    
                S = S_next # Update the state info
                if epsilon > 0.1: # Keep some exploration
                    epsilon -= decay_rate # Reduce epsilon as policy learns
                
                if iters % check_rate == 0:
                    # Test the model performance
                    test_reward,_ = self.test(episodes=20, epsilon=0.05) # Run a test to check the performance of the model
                    print('Reward: {}, Step: {}'.format(test_reward, tf.train.global_step(self.net.sess, self.net.global_step)))
                    reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Test_Reward', simple_value=test_reward)])
                    self.net.writer.add_summary(reward_summary, tf.train.global_step(self.net.sess, self.net.global_step))
                    done = True
                iters += 1
            
            ep_reward_summary = tf.Summary(value=[tf.Summary.Value(tag='Episode_Reward', simple_value=ep_reward)])
            self.net.writer.add_summary(ep_reward_summary, tf.train.global_step(self.net.sess, self.net.global_step))
            if ep % 50 == 0:
                print("episode {} complete, epsilon={}".format(ep, epsilon))
            if ep % 250 == 0  and ep != 0:
                self.net.save_model_weights()

    def test(self, model_file=None, episodes=100, epsilon=0.0):
        # Evaluate the performance of the agent over 100 episodes
        total_reward = 0
        rewards = []
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
            rewards += episode_reward
        
        average_reward = float(total_reward/episodes)
        return average_reward, rewards
    
    def burn_in_memory(self, memory_queue, burn_in=10000):
        # Initialize the replay memory with a burn_in number of episodes / transitions. 
        i = 0
        while True:
            S = self.env.reset()
            done = False
            while not done:
                if i >= burn_in:
                    return
                features = self.net.getFeatures(S)
                # Epsilon greedy training policy
                if np.random.sample() < 0.5: # Add some stochasticity to the burn in
                    action = self.env.action_space.sample()
                else:
                    q_vals = self.net.infer(features)
                    action = np.argmax(q_vals)
                # Execute selected action
                S_next, R, done,_ = self.env.step(action)
                if self.linear:
                    features = features[None,action,:] # (1, state*action)
                feature_next = self.net.getFeatures(S_next)
                store = (features, action, R, done, feature_next)
                memory_queue.append(store)
                S = S_next
                i += 1
        

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

    # Create an instance of the environment and agent
    env = gym.make(environment_name)
    agent = DQN_Agent(env, sess=sess, network_type=network_type)
    
    agent.render = True
    for i in range(5):
        agent.train()
        reward = agent.test(episodes=20)
        print("Test reward: {}".format(reward))
    env.close()
    

if __name__ == '__main__':
    main(sys.argv)

