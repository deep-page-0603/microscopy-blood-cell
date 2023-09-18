
import tensorflow as tf
import numpy as np
import joblib
from common import *
from board import Board
import random

""" Hilord Core Neural Network """

class NN(object):
    def __init__(self, name = 'Hilord', filename = None):
        self.sess = tf.Session()
        self.name = name

        self.lr_multiplier = 1

        # All placeholders
        self.state_ex = tf.placeholder(tf.float32, [None, NN_EX_DIM], 'state_ex')
        self.reward = tf.placeholder(tf.float32, [None, 1], 'reward')

        # Critic
        self.pi = self._policy_net(self.name + '/pi')
        self.v_off = self.reward - self.pi
        self.c_loss = tf.reduce_mean(tf.square(self.v_off))
        self.c_opt = tf.train.AdamOptimizer(NN_C_LR * self.lr_multiplier).minimize(self.c_loss)

        self.sess.run(tf.global_variables_initializer())
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)

        if filename: self.restore(filename)

    def update(self, queue):
        random.shuffle(queue)

        #while len(queue) > 0:
        for i in range(1):
            #subQue = queue[:10000]
            #queue = queue[10000:]
            subQue = queue

            data = np.vstack(subQue)
            state_ex = data[:, :NN_EX_DIM]
            reward = data[:, -1:]
            #old_prob = self.sess.run(self.pi, {self.state_ex : state_ex})[:, 0]

            for _ in range(NN_UPDATE_STEP):
                self.sess.run([self.c_opt, self.c_loss], {
                    self.state_ex: state_ex, self.reward: reward})

        #return epoch, kl, self.lr_multiplier
        return 0, 0, 0

    def _policy_net(self, name):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.state_ex, 256, tf.nn.relu)
            l2 = tf.layers.dense(l1, 512, tf.nn.relu)
            l3 = tf.layers.dense(l2, 1024, tf.nn.relu)
            l4 = tf.layers.dense(l3, 2048, tf.nn.relu)
            l5 = tf.layers.dense(l4, 4096, tf.nn.relu)
            l6 = tf.layers.dense(l5, 2048, tf.nn.relu)
            l7 = tf.layers.dense(l6, 1024, tf.nn.relu)
            l8 = tf.layers.dense(l7, 512, tf.nn.relu)
            l9 = tf.layers.dense(l8, 256, tf.nn.relu)
            act_out = tf.layers.dense(l9, 1, None)
        return act_out

    def choose_action(self, state_ex, is_train, is_agent = False):
        pi = self.sess.run(self.pi, {self.state_ex : state_ex})[:, 0]

        if is_train and (is_agent or not ARGMAX_MODE):
            return np.argmax(pi)
            #return np.random.choice(list(range(len(pi))), p = softmax(pi))
            # if random.randint(0, 9) >= 3:
            #     return np.argmax(pi)
            # else:
            #     return np.random.choice(list(range(len(pi))), p = softmax(pi))
        else:
            return np.argmax(pi)

    def get_variables(self, net_scope, depth):
        res = []
        for i in range(depth):
            scope = self.name + '/' + net_scope + '/dense'
            
            if i > 0: scope += '_' + str(i)
                
            with tf.variable_scope(scope, reuse = True):
                k = tf.get_variable('kernel')
                b = tf.get_variable('bias')
            res.append(self.sess.run([k, b]))
        return res

    def get_model(self):
        return self.sess.run(self.params)
    
    def load_model(self, saved_params):
        self.sess.run([p.assign(sp) for p, sp in zip(self.params, saved_params)])        
    
    def save(self, filename):
        joblib.dump(self.get_model(), filename)

    def restore(self, filename):
        self.load_model(joblib.load(filename))

if __name__ == '__main__':
    net = NN('port', './model/best_farmer_net.model')
    port_net_to_txt(net, 'pi', NN_POLICY_DEPTH, 'd:/farmer_net_vars.cpp')