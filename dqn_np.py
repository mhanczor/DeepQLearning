#!/usr/bin/env python
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

        self.env = environment
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        
    
    def save_model_weights(self, suffix):
        # Helper function to save your model / weights. 
        pass

    def load_model(self, model_file):
        # Helper function to load an existing model.
        pass

    def load_model_weights(self,weight_file):
        # Helper funciton to load model weights. 
        pass
        
        
class LinearQ(QNetwork):
    
    def __init__(self, environment, alpha=0.0001, gamma=1.0, filepath='tmp/linearq/'):
        # Initialize weight matrics
        super(LinearQ, self).__init__(environment, alpha, gamma)
        
        self.filepath=filepath
        self.nA =self.env.action_space.n
        self.inputs = self.env.observation_space.shape[0]
        
        # Weights and inputs can be defined as tensorflow or numpy vectors:
        # Initializing weights
        np.random.seed(2) # DON'T TOUCH!! THIS SEED WORKS! # Epsilon = 0.7
        self.w = np.random.normal(scale=2.0, size=(self.nA*self.inputs+1, 1))
        self.lam = 0.001
        
    def evaluate(self, S, a):
        fv = self._getFeatures(S,a)

        return np.dot(self.w.T, fv) # linear q approximation
        
    def update(self, S, a, S_next, R, terminal=False):
        fv = self._getFeatures(S, a)
        q = self.evaluate(S, a)
        q_values = np.zeros((self.nA, 1))
        if terminal:
            self.w += self.alpha*(R - q)*fv
        else:
            # Evaluate over the next state
            # Update the parameters of the linear approximator
            for i in range(self.nA):
                q_values[i] = self.evaluate(S_next, i)
            max_q = np.max(q_values)
            # loss = (R + self.gamma*max_q - q) + self.lam*np.dot(self.w.T, self.W)
            self.w = self.w + self.alpha*((R + self.gamma*max_q - q)*fv - self.lam*self.w) # includes regularization
    
    def _getFeatures(self, S, a):
        # Create feature vector
        # Feature vector is a combination of the inputs and selected action
        
        S = np.atleast_2d(S).T # Column vector
        A = np.zeros((self.nA, 1)) # Action is zero-index, create one hot vector
        A[a] = 1
        # Create a feature vector from the actions and state info
        fv = np.zeros((self.nA*self.inputs+1, 1))
        fv[1:] = (np.dot(S, A.T)).reshape((self.nA*self.inputs,1))
        fv[0] = 1 # for bias
        
        return fv
        
    def save_model_weights(self):
        np.save(self.filepath + 'model.npy', self.w)
        print(self.w)
        print('Saved weights to {}model.npy'.format(self.filepath))
    
    def load_model_weights(self):
        self.w = np.load(self.filepath + 'model.npy')
        print(self.w)
        print('Loaded weights from {}model.npy'.format(self.filepath))
                

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
    
    def __init__(self, environment, network_type, epsilon=0.7, render=False):

        self.env = environment
        self.nA = self.env.action_space.n
        self.render = render
        self.epsilon = epsilon
                
        if network_type == 'Linear':
            self.net = LinearQ(environment)
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
                # Evaluate the q values for each action at the current state
                q_values[i] = self.net.evaluate(S, i)
            
            action = self.epsilon_greedy_policy(q_values) #Select an actions based on the epsilon greedy policy
            newS, R, done,_ = self.env.step(action) # Take the next step
            self.net.update(S, action, newS, R, done) # Update the model
            iters += 1
            if done:
                S = self.env.reset()
                cur_eps += 1
                if cur_eps % 100 == 0:
                    print('{} episodes complete, {} iterations'.format(cur_eps, iters))
            else:
                S = newS
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
                q_values[i] = self.net.evaluate(S, i)
            
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
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # You want to create an instance of the DQN_Agent class here, and then train / test it.
    env = gym.make(environment_name)
    agent = DQN_Agent(env, network_type)
    
    
    # agent.net.save_model_weights('test_name')
    
    agent.render = True
    for i in range(5):
        agent.train()
        agent.test(episodes=4)
    env.close()
    

if __name__ == '__main__':
    main(sys.argv)

