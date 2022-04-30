import numpy as np
from collections import defaultdict
import itertools

from fee_function import fee



class matrix:
    def __init__(self, k, a):
        self.nodes = defaultdict(dict)
        self.length = k+1
        self.adv_prob = a
        for h, a in itertools.product(range(self.length), range(self.length)):
            self.nodes[h][a] = node(h, a)
            
    def display(self, index=False):
        for h in range(self.length):
            for a in range(self.length):
                self.nodes[h][a].display(index)
                print('\t|', end='')
            print()
            
    def compute_probabilities(self):
        if self.adv_prob < 0 or self.adv_prob > 1.:
            print("Please assign adversary hash power to be a number between 0 and 1")
            return

        honest_prob = 1. - self.adv_prob
        
        for h in range(self.length):
            self.nodes[h][0].set_prob(honest_prob**h)
        
        for a in range(self.length):
            self.nodes[0][a].set_prob(self.adv_prob**a)
        
        for h in range(1, self.length):
            for a in range(1, self.length):
                p = self.nodes[h-1][a].get_prob() * honest_prob + self.nodes[h][a-1].get_prob() * self.adv_prob
                self.nodes[h][a].set_prob(p)
                
    def adversary_success_prob(self):
        p = 0.
        a = self.length
        for h in range(a):
            p += self.nodes[h][a-1].get_prob() * self.adv_prob
            
        return p
    
    def adversary_expected_blocks_attempted(self):
        b = 0.
        a = self.length
        for h in range(a):
            p = self.nodes[h][a-1].get_prob() * self.adv_prob
            b += p * (h + a)
        
        b += (1 - self.adversary_success_prob()) * (2 * self.length - 1)
        
        return b
                                              
class node:
    def __init__(self, h_blocks, a_blocks):
        self.h_blocks = h_blocks
        self.a_blocks = a_blocks
        self.prob = 0.
        
    def display(self, index=False):
        if index:
            print('H: ' + str(self.h_blocks) + ' , ' + 'A: ' + str(self.a_blocks) + ' : ' + str(self.prob), end='')
        else:
            print(str(self.prob), end='')
        
    def set_prob(self, p):
        self.prob = p
        
    def get_prob(self):
        return self.prob
    
    
def prob_success(k, beta, verbose=False):
    m = matrix(k, beta)
    m.compute_probabilities()
    if verbose:
        m.display(True)
    
    return m.adversary_success_prob()

def blocks_attempted(k, beta, verbose=False):
    m = matrix(k, beta)
    m.compute_probabilities()
    if verbose:
        m.display(True)
        
    return m.adversary_expected_blocks_attempted()

# the probability that an adversary succeeds in their attack
def adv_prob(alpha_bar, beta, k): 
    return prob_success(k, beta)

# the expected cost an adversary incurs mounting an attack
def adv_cost(alpha_bar, beta, k, block_reward, num_agents, mining_cost):
    adv_cost_per_block = beta * alpha_bar * num_agents * mining_cost
    return blocks_attempted(k, beta) * adv_cost_per_block
    
# the expected reward an adversary gains from a successful attack
def adv_reward(z, alpha_bar, beta, k, block_reward):
    return ((k+1)*block_reward + z + fee(z))*adv_prob(alpha_bar, beta, k)

# returns 1 if the adversary's expected reward is higher than their expected cost and 0 otherwise
def A(z, alpha_bar, beta, k, block_reward, num_agents, mining_cost):
    r = adv_reward(z, alpha_bar, beta, k, block_reward) - adv_cost(alpha_bar, beta, k, block_reward, num_agents, mining_cost)
    return 1 / (1 + np.exp(-r))