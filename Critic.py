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


LEARNING_RATE = 0.002
BATCH_SIZE = 100
hidden1 = 400
hidden2 = 300

class Critic(object):

    def __init__(self, sess, state_dim, action_dim, tau, num_actor_vars):

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.num_actor_vars = num_actor_vars

        self.inputs, self.action, self.out = self._build_network()
        self.main_parameters = tf.trainable_variables()[self.num_actor_vars : ]

        self.target_inputs, self.target_action, self.target_out = self._build_network()
        self.target_parameters = tf.trainable_variables()[len(self.main_parameters) + self.num_actor_vars : ]

        # weight update graph 생성
        self.update_target_params = [self.target_parameters[i].assign(tf.multiply(self.main_parameters[i], self.tau) + tf.multiply(self.target_parameters[i], 1. - self.tau)) for i in range(len(self.target_parameters))]


        # actor network 가 current policy를 기반으로 예측한 q value 값을 받아온다.
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # critic 은 정책 (policy)의 평가자이므로, actor network가 가진 current policy의 결과값을 가져와서
        # critic이 가진 network의 policy와 비교하여 action-value function을 학습한다.
        # actor network의 예측값과 critic network의 parameter vector "w" 에 의해서계산된 output 값의 차이가 loss function이 된다.
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predicted_q_value, self.out))))

        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

        # Q(s, a) da 의 정보를 넘겨줘야 하므로, 다음과 같이 out을 action에  대해서 미분 한다.
        self.action_grads = tf.gradients(self.out, self.action)


    def _build_network(self):

        # Q(s,a) action-value function을 대체하는 네트워크 이므로,
        # input으로 state와 action이 들어온다.
        inputs = tflearn.input_data(shape=[None, self.state_dim])
        action = tflearn.input_data(shape=[None, self.action_dim])

        self.w1 = weight_var([self.state_dim, hidden1], 'weight/layer1/critic')
        b1 = bias_var([hidden1], 'weight/layer1/critic')

        w2 = weight_var([hidden1, hidden2], 'weight/layer2/critic')
        b2 = bias_var([hidden2], 'bias/layer2/critic')

        w3 = weight_var([hidden2, 1], 'weight/layer3/critic')
        b3 = bias_var([1], 'bias/layer3/critic')

        w2_action = weight_var([self.action_dim, hidden2], 'weight/layer2/action_critic')

        h1 = tf.nn.relu(tf.matmul(inputs, self.w1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + tf.matmul(action, w2_action) + b2)
        out = tf.matmul(h2, w3) + b3

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

    def update(self):
        self.sess.run(self.update_target_params)


def weight_var(shape, name):
    return tf.Variable(tf.truncated_normal(shape = shape, stddev=0.01), name = name)

def bias_var(shape, name):
    return tf.Variable(tf.constant(0.03), name = name)