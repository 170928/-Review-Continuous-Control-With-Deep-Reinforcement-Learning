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
    def __init__(self, sess, state_dim, action_dim, action_bound, tau, action_type):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.tau = tau
        self.action_type = action_type


        # Parameters "soft update" is needed.
        # We have 2 actor networks.

        # mainActorNetwork
        self.inputs, self.phase, self.outputs, self.scaled_outputs = self._build_actor_network()
        self.main_parameters = tf.trainable_variables()

        # targetActorNetwork
        self.target_inputs, self.target_phase, self.target_outputs, self.target_scaled_outputs = self._build_actor_network()
        self.target_parameters = tf.trainable_variables()[len(self.main_parameters):]

        # parameter "soft" update
        self.update_target_net_params = \
            [self.target_parameters[i].assign(tf.multiply(self.main_parameters[i], self.tau) +
                                              tf.multiply(self.target_parameters[i], 1. - self.tau))
             for i in range(len(self.target_parameters))]

        # tf.gradients(y, x, grad_ys) = grad_ys * diff(y, x)
        # tf.gradients()는 백프롭 할 때 미분된 값을 보여주고, apply_gradients()는 .minimize(loss)와 같은 역할을 하지만 직접 그래디언트 값을 넣어서 theta를 바꿀 수 있습니다.
        # tf.gradients() 를 사용해 미분을 수행해 기울기(gradient)를 구합니다. 더 정확히는 X(들)에 대한 Y(들)의 편미분값을 구해 줍니

        if self.action_type == 'Continuous':
            self.action_gradients = tf.placeholder(tf.float32, [None, self.action_dim])
        else:
            self.action_gradients = tf.placeholder(tf.float32, [None, 1])

        self.actor_gradients = tf.gradients(self.outputs, self.main_parameters, -self.action_gradients)

        # Optimization Op
        # apply_gradients 에 대한 내용 참조
        # http://dukwon.tistory.com/31
        # actor_gradients 는 self.main_parameters에 대한 gradients가 들어있다.
        # main_parameters에 대한 gradients를 이용해서
        # main_parameters를 update 해야하므로,
        # optimizer의 apply_gradients를 사용한다.
        # tf.gradients 와 appliy_gradients 를 사용하는 이유는 collect the gradients and apply later 를 하고자 할 때입니다.
        # 이렇게 나누는 방법의 쉬운 예로는 loss 를 구하고 해당 loss 값에 의한 gradients를 특정 범위로 clip 후에 학습하고자 할때 사용되며
        # https://medium.com/@dubovikov.kirill/actually-we-can-work-with-gradients-directly-in-tensorflow-via-optimizers-compute-gradients-and-fc2b5612665a
        # 위의 reference에서 이해할 수 잇습니다.

        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE)
        self.grads_and_vars = list(zip(self.actor_gradients, self.main_parameters))
        self.updateGradients = self.optimize.apply_gradients(self.grads_and_vars)

        self.num_trainable_vars = len(self.main_parameters) + len(self.target_parameters)


    def _build_actor_network(self):



    
