# [Reference Code]
# https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py
# https://github.com/slowbull/DDPG


import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp

from replay_buffer import ReplayBuffer


LEARNING_RATE = 0.00025
BATCH_SIZE = 64

class Actor(object):
    def __init__(self, sess, state_dim, action_dim, action_bound):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound


        # Parameters "soft update" is needed.
        # We have 2 actor networks.
        # mainActorNetwork


        # tf.trainable_variables() << -- calling position is important
        self.main_parameters = tf.trainable_variables()

        # targetActorNetwork

        self.target_parameters = tf.trainable_variables()[len(self.main_parameters):]


    
