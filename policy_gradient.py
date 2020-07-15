"""
Policy Gradient Reinforcement Learning
Uses a 3 layer neural network as the policy network

"""
import math
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import random

'''
def uniform_distribution(a):
    l = len(a)
    index = random.randint(0,l-1)
    return a[index]
'''


def cascade_model(p,k):
    return 1-(1-p)**k


def gradient_cascade(p, k):
    return k*(1-p)**(k-1)


class PolicyGradient:
    def __init__(
        self,
        n_x,
        n_y,
        s0,
        learning_rate=0.01,
        reward_decay=0.95,
        load_path=None,
        save_path=None,
        weight_capping_c=math.e**3,
        k=2,
        b_distribution='uniform'
    ):

        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.gamma = reward_decay

        self.save_path = None
        if save_path is not None:
            self.save_path = save_path
        '''
        weight_capping_c cap the coefficient to reduce variance 
        '''
        self.weight_capping_c = weight_capping_c

        # num of items on the slate
        self.K = k
        self.s0 = s0
        self.b_distribution = b_distribution
        self.episode_observations = [s0]
        self.episode_actions, self.episode_rewards = [], []

        self.sess = tf.Session()
        self.cost_history = []

        # print(self.n_x,self.n_y)
        self.build_network()

        # $ tensorboard --logdir=logs
        # http://0.0.0.0:6006/
        self.f_summary = tf.summary.FileWriter("logs/", self.sess.graph)

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

        # Restore model
        if load_path is not None:
            self.load_path = load_path
            self.saver.restore(self.sess, self.load_path)

    def weight_capping(self,cof):
        return min(cof,self.weight_capping_c)

    def store_transition(self, s, a, r):
        """
            Store play memory for training

            Arguments:
                s: observation
                a: action taken
                r: reward after action
        """
        self.episode_observations.append(s)
        self.episode_rewards.append(r)

        # Store actions as list of arrays
        # e.g. for n_y = 2 -> [ array([ 1.,  0.]), array([ 0.,  1.]), array([ 0.,  1.]), array([ 1.,  0.]) ]
        action = np.zeros(self.n_y)
        action[a] = 1
        self.episode_actions.append(action)

    def uniform_choose_action(self,observation):

        # Reshape observation to (num_features, 1)
        observation = observation[:, np.newaxis]

        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict={self.X: observation})

        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(range(len(prob_weights.ravel())), p=[1/len(prob_weights.ravel())]*(len(prob_weights.ravel()))
)
        return action

    def choose_action(self, observation):
        """
            Choose action based on observation

            Arguments:
                observation: array of state, has shape (num_features)

            Returns: index of action we want to choose
        """
        # Reshape observation to (num_features, 1)
        # print(observation)
        observation = np.array(observation)
        observation = observation[:, np.newaxis]
        # print(observation)
        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict = {self.X: observation})

        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())

        # exploration to allow rare data appeared
        if random.randint(0,1000) < 1000:
            pass
        else:
            action = random.randint(0,self.n_y-1)
        return action

    def behaviour_action(self,obervation,dirt = 'uniform'):
        if dirt == 'uniform':
            action = np.random.choice(range(self.n_y))
        else:
            prob = [i/55 for i in range(1,self.n_y+1)]
            action = np.random.choice(range(self.n_y), p=prob)
        return action

    def learn(self):
        # Discount and normalize episode reward
        discounted_episode_rewards_norm = self.discount_and_norm_rewards()

        # Train on episode
        self.sess.run(self.train_op, feed_dict={
             self.X: np.vstack(self.episode_observations[:-1]).T,
             self.Y: np.vstack(np.array(self.episode_actions)).T,
             self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
        })
        '''
        cost = self.sess.run(self.loss, feed_dict={self.X: np.vstack(self.episode_observations[:-1]).T,
                                                   self.Y: np.vstack(np.array(self.episode_actions)).T,
                                                   self.discounted_episode_rewards_norm: discounted_episode_rewards_norm})
        '''
        #self.cost_history.append(discounted_episode_rewards_norm)
        # Reset the episode data
        self.episode_observations, self.episode_actions, self.episode_rewards  = [self.s0], [], []

        # Save checkpoint
        #if self.save_path is not None:
        #    save_path = self.saver.save(self.sess, self.save_path)
        #    print("Model saved in file: %s" % save_path)

        return discounted_episode_rewards_norm

    def discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards,dtype='float64')
        cumulative = 0
        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = cumulative
        # Normalize the rewards
        #discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        #discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards


    def build_network(self):
        # Create placeholders
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(self.n_x, None), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(self.n_y, None), name="Y")
            self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # Initialize parameters
        units_layer_1 = 10
        units_layer_2 = 10
        units_output_layer = self.n_y
        with tf.name_scope('parameters'):
            W1 = tf.get_variable("W1", [units_layer_1, self.n_x], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b1 = tf.get_variable("b1", [units_layer_1, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W2 = tf.get_variable("W2", [units_layer_2, units_layer_1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b2 = tf.get_variable("b2", [units_layer_2, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W3 = tf.get_variable("W3", [self.n_y, units_layer_2], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            b3 = tf.get_variable("b3", [self.n_y, 1], initializer = tf.contrib.layers.xavier_initializer(seed=1))

        # Forward prop
        with tf.name_scope('layer_1'):
            Z1 = tf.add(tf.matmul(W1,self.X), b1)
            A1 = tf.nn.relu(Z1)
        with tf.name_scope('layer_2'):
            Z2 = tf.add(tf.matmul(W2, A1), b2)
            A2 = tf.nn.relu(Z2)
        with tf.name_scope('layer_3'):
            Z3 = tf.add(tf.matmul(W3, A2), b3)
            A3 = tf.nn.softmax(Z3)

        self.sess.run(tf.global_variables_initializer())

        # Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
        logits = tf.transpose(Z3)
        labels = tf.transpose(self.Y)
        self.outputs_softmax = tf.nn.softmax(logits, name='A3')
        self.generate_softmax = tf.nn.softmax(logits, name='A3')


        with tf.name_scope('loss'):

            l = self.outputs_softmax.shape.as_list()[1]
            #print(l)
            #print(self.episode_observations[-1])
            observation = np.array(self.episode_observations[-1])
            at = self.behaviour_action(observation,self.b_distribution)
            #print(s)
            observation.shape = (len(observation),1)
            #print(s)
            prob_weights = self.sess.run(self.generate_softmax, feed_dict={self.X: observation})
            print(prob_weights)
            p_at = prob_weights[0][at]
            print(p_at)
            # off-policy correction,a/b,here behavior policy is uniform distribution
            # induce weight capping to reduce variance
            off_policy_correction = self.weight_capping(p_at * l)
            # top-k correction
            topk_correction = gradient_cascade(p_at, self.K)
            # log gradient
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            self.loss = tf.reduce_sum(neg_log_prob * self.discounted_episode_rewards_norm
                                  * topk_correction * off_policy_correction)  # reward guided loss


        with tf.name_scope('train'):
            #self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    # return trained distribution
    def get_distribution(self,s):
        s = np.array(s)
        s.shape = (len(s), 1)
        prob_weights = self.sess.run(self.generate_softmax, feed_dict={self.X: s})
        return prob_weights[0]

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Reward')
        plt.xlabel('Training Steps')
        plt.show()
