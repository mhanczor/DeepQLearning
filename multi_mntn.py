#!/usr/bin/env python
import tensorflow as tf
import gym
from DQN_Implementation import DQN_Agent
import numpy as np

mean_max = 0
best_example = ''
for i in range (100):
    tf.reset_default_graph()
    env = gym.make('MountainCar-v0')
    episodes=1e3
    decay_rate=1e-6

    filepath = 'tmp/mountaincar/run' + str(i) + '/'
    # Dueling DeepQ Network
    gamma = 1.0
    alpha = 0.0001
    epsilon = 0.5
    network = 'DDNN' # Dueling Network
    replay = True
        
    with tf.Session() as sess:
        # Initialize agent
        tf.set_random_seed(i)
        agent = DQN_Agent(environment=env, 
                            sess=sess, 
                            network_type=network,
                            gamma=gamma,
                            filepath=filepath,
                            alpha=alpha,
                            double=True)

        # Train the network
        agent.train(episodes=episodes,
                    epsilon=epsilon,
                    decay_rate=decay_rate,
                    replay=replay)
        agent.net.save_model_weights()
                    
        # Final testing
        agent.render = False
        total_reward, rewards = agent.test(episodes=100, epsilon=0.05)
        rewards = np.array(rewards)
        std_dev = np.std(rewards)
        mean = np.mean(rewards)
        print("MountainCar Mean: {}, StdDev: {}".format(mean, std_dev))
        if mean > mean_max:
            mean_max = mean
            best_example = filepath

        agent.net.writer.close()
        agent.env.close()
        
print("Best mountaincar run was {}".format(best_example))



