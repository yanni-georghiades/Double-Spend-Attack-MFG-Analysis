import numpy as np
import scipy as sp
from scipy.stats import binom
from matplotlib import pyplot as plt
import sys

from output import Experiment
from helper_functions import select_best_parameters, best_actions
from environment_model import max_wealth, max_tx_value, beta_default, block_reward_default, \
                                mining_cost_default, num_agents_default, \
                                k_default, reward, win_probability, win_reward


VALUE_HISTORY = []
ALPHA = []
WEALTH_HISTORY = []
ALPHA_BAR = []
Z = []
T = 5
N = 1
momentum = .1
alpha_bar_init = 5.


def main():

    experiment = Experiment(k_default, beta_default, block_reward_default, alpha_bar_init, mining_cost_default, 
                        num_agents_default, max_wealth, max_tx_value, T, N, momentum)

    ab = [5.] * (T + 1)
    ALPHA_BAR.append(ab)

    for n in range(N):
        vx = []
        ax = []
        zx = []
        for x in range(max_wealth):
            alpha, z = select_best_parameters(0, x, 0, max_tx_value, 5, ALPHA_BAR[n][0], beta_default, 
                                          k_default, mining_cost_default, num_agents_default, 
                                          block_reward_default, print_summary=False)
            rew = reward(alpha, z, ALPHA_BAR[n][0])
            ax.append(alpha)
            zx.append(z)
            vx.append(rew)
            print(z)
        
        ALPHA.append(ax)
        Z.append(zx)
        
        VALUE = []
        VALUE.append(vx)

        for t in range(T, 0, -1):
            vx = []
            ax = []
            zx = []
            for wealth in range(max_wealth):
                alpha, z, val = best_actions(ALPHA_BAR[n][t], wealth, VALUE[T-t])
                ax.append(alpha)
                zx.append(z)
                vx.append(val)
                
            ALPHA.append(ax)
            Z.append(zx)
            VALUE.append(vx)
        
        VALUE_HISTORY.append(VALUE)
        
        # initialize wealth randomly
        WEALTH = []
        w = np.random.rand(max_wealth, )
        wealth_distribution = w / sum(w)
        WEALTH.append(wealth_distribution)
        
        for t in range(0, T+1, 1):
            wealth_distribution = np.zeros(max_wealth)
            for wealth in range(max_wealth):
                alpha = ALPHA[t][wealth]
                z = Z[t][wealth]
            
        
    #             The following is code which factors in probability of win/loss instead of simply expected reward
                wp = win_probability(alpha, ALPHA_BAR[n][t], num_agents_default)
                wr = win_reward(z, ALPHA_BAR[n][t], beta_default, k_default, mining_cost_default, 
                                num_agents_default, block_reward_default) - alpha * mining_cost_default
                lr = - alpha * mining_cost_default
                
                win_wealth = round(wealth + wr)
                lose_wealth = round(wealth + lr)
                if lose_wealth < 0:
                    print("negative wealth detected")
                wealth_distribution[min(win_wealth, max_wealth) - 1] += WEALTH[t][wealth] * wp
                wealth_distribution[min(lose_wealth, max_wealth) - 1] += WEALTH[t][wealth] * (1 - wp)
                
                WEALTH.append(wealth_distribution)
                
        # save wealth distribution from time T
        WEALTH_HISTORY.append(WEALTH)
            
        alpha_bar_next = []
        for t in range(0, T+1):
            avg = 0.
            for wealth in range(max_wealth):
                alpha = ALPHA[t][wealth]
                density = WEALTH[t][wealth]
                avg += alpha * density
            
            new_alpha_bar = momentum * ALPHA_BAR[n][t] + (1 - momentum) * avg
            alpha_bar_next.append(new_alpha_bar)
            
        ALPHA_BAR.append(alpha_bar_next)

    experiment.add_results(VALUE_HISTORY, ALPHA, WEALTH_HISTORY, ALPHA_BAR, Z)
    experiment.save_to_file(sys.argv[1])



if __name__ == "__main__":
    main()