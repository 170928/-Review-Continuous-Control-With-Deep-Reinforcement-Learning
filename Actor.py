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

class Actor(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, tau):

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.tau = tau

        self.hidden1 = 400
        self.hidden2 = 300


        # Parameters "soft update" is needed.
        # We have 2 actor networks.

        # mainActorNetwork
        self.inputs, self.outputs, self.scaled_outputs = self._build_actor_network()
        self.main_parameters = tf.trainable_variables()

        # targetActorNetwork
        self.target_inputs, self.target_outputs, self.target_scaled_outputs = self._build_actor_network()
        self.target_parameters = tf.trainable_variables()[len(self.main_parameters):]

        # parameter "soft" update
        self.update_target_params = [
                                        self.target_parameters[i].assign(tf.multiply(self.main_parameters[i], self.tau) + tf.multiply(self.target_parameters[i], 1. - self.tau))
                                        for i in range(len(self.target_parameters))
                                    ]

        # tf.gradients(y, x, grad_ys) = grad_ys * diff(y, x)
        # tf.gradients()는 백프롭 할 때 미분된 값을 보여주고, apply_gradients()는 .minimize(loss)와 같은 역할을 하지만 직접 그래디언트 값을 넣어서 theta를 바꿀 수 있습니다.
        # tf.gradients() 를 사용해 미분을 수행해 기울기(gradient)를 구합니다. 더 정확히는 X(들)에 대한 Y(들)의 편미분값을 구해 줍니

        # Critic network가 전달해주는 action-value function의 gradient값이 들어올 ph 를 만듭니다.
        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_dim])

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
        self.grads_and_vars = list(zip(self.actor_gradients, self.main_parameters))
        self.updateGradients = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(self.grads_and_vars)

        #self.optimize = tf.train.AdamOptimizer(LEARNING_RATE). \
        #    apply_gradients(zip(self.actor_gradients, self.main_parameters))



        self.num_trainable_vars = len(self.main_parameters) + len(self.target_parameters)


    def _build_actor_network(self):

        inputs = tf.placeholder(tf.float32, [None, self.state_dim])

        w1 = weight_var( [self.state_dim, self.hidden1] ,'hidden1/weight')
        b1 = bias_var( [self.hidden1], 'hidden1/bias')

        w2 = weight_var( [self.hidden1, self.hidden2] , 'hidden2/weight')
        b2 = bias_var( [self.hidden2], 'hidden2/bias')

        w3 = weight_var( [self.hidden2, self.action_dim], 'hidden3/weight')
        b3 = bias_var( [self.action_dim], 'hidden3/bias')

        h1 = tf.matmul(inputs, w1) + b1
        h1 = tf.nn.relu(h1)

        h2 = tf.matmul(h1, w2) + b2
        h2 = tf.nn.relu(h2)

        # output을 -1~1 로 매핑 시키기 위해서 tanh를 사용합니다.
        outputs = tf.nn.tanh(tf.matmul(h2, w3) + b3)

        # output의 scale을 +- action_bound로 만들기 위해서 out에 action_bound를 곱합니다
        scaled_outputs = tf.multiply(outputs, self.action_bound)

        return inputs, outputs, scaled_outputs

    def train(self, inputs, action_grad):
        self.sess.run(self.updateGradients, feed_dict={self.inputs : inputs, self.action_gradients : action_grad})

    def predict(self, inputs):
        return self.sess.run(self.scaled_outputs, feed_dict = {self.inputs : inputs})

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_outputs, feed_dict = {self.target_inputs : inputs})

    def update(self):
        # soft weight update graph 를 실행합니다.
        self.sess.run(self.update_target_params)

    def get_num_vars(self):
        return self.num_trainable_vars



def weight_var(shape, name):
    return tf.Variable(tf.truncated_normal(shape = shape, stddev=0.01), name = name)

def bias_var(shape, name):
    return tf.Variable(tf.constant(0.03), name = name)