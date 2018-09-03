# [Reference Code]
# https://github.com/pemami4911/deep-rl/blob/master/ddpg/ddpg.py
# https://github.com/slowbull/DDPG
# https://github.com/liampetti/DDPG

import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp

from replay_buffer import ReplayBuffer


LEARNING_RATE = 0.00025
BATCH_SIZE = 100

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, tau, num_actor_vars):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.tau = tau
        self.num_actor_vars = num_actor_vars

        self.inputs, self.action, self.out = self._build_network()
        self.main_parameters = tf.trainable_variables()[self.num_actor_vars:]

        self.target_inputs, self.target_action, self.target_out = self._build_network()
        self.target_parameters = tf.trainable_variables()[len(self.main_parameters) + self.num_actor_vars : ]

        # weight update graph 생성
        self.update_target_params = [self.target_parameters[i].assign(tf.multiply(self.main_parameters[i], self.tau) + tf.multiply(self.target_parameters[i], 1. - self.tau)) for i in range(len(self.target_parameters))]

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
        self.loss = tflean.mean_square(self.predicted_q_value, self.out)

        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

        self.action_grads = tf.gradients(self.out, self.action)



    def _build_network(self):
        inputs = tflearn.input_data(shape=[None, self.state_dim])
        action = tflearn.input_data(shape=[None, self.action_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # output 은 critic 의 결과인 action-value function 이므로 Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)

        return inputs, action, out


    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_params)
