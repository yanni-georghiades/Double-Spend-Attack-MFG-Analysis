import numpy as np
import scipy as sp
from scipy.stats import binom
from matplotlib import pyplot as plt
import sys

from output import Experiment
from helper_functions import  best_actions
from environment_model import reward, win_probability, win_reward


def main():

    exp = Experiment(   k=6, 
                        beta=.4,
                        block_reward=10,
                        alpha_bar_init=1.,
                        mining_cost=1,
                        num_agents=10,
                        max_wealth=100,
                        max_tx_value=100,
                        T=4,
                        N=4,
                        momentum=.9)

    VALUE_HISTORY = []
    ALPHA_HISTORY = []
    WEALTH_HISTORY = []
    ALPHA_BAR_HISTORY = []
    Z_HISTORY = []

    ALPHA_BAR = [exp.alpha_bar_init] * (exp.T + 1)
    ALPHA_BAR_HISTORY.append(ALPHA_BAR)

    for n in range(exp.N):
        Z = []
        ALPHA = []
        VALUE = []

        vx = []
        ax = []
        zx = []
        for wealth in range(exp.max_wealth):
            alpha, z, val = best_actions(exp, ALPHA_BAR_HISTORY[n][0], wealth, [0] * exp.max_wealth)
            rew = reward(exp, alpha, z, ALPHA_BAR_HISTORY[n][0])
            ax.append(alpha)
            zx.append(z)
            vx.append(rew)
        
        ALPHA.append(ax)
        Z.append(zx)
        VALUE.append(vx)

        for t in range(exp.T, 0, -1):
            vx = []
            ax = []
            zx = []
            for wealth in range(exp.max_wealth):
                alpha, z, val = best_actions(exp, ALPHA_BAR_HISTORY[n][exp.T-t], wealth, VALUE[exp.T-t])
                ax.append(alpha)
                zx.append(z)
                vx.append(val)
                
            ALPHA.append(ax)
            Z.append(zx)
            VALUE.append(vx)
        
        # reverse lists in place because we constructed them backwards
        VALUE.reverse()
        Z.reverse()
        ALPHA.reverse()

        VALUE_HISTORY.append(VALUE)
        Z_HISTORY.append(Z)
        ALPHA_HISTORY.append(ALPHA)
        
        # initialize wealth randomly
        WEALTH = []
        w = np.random.rand(exp.max_wealth, )
        wealth_distribution = w / sum(w)
        WEALTH.append(wealth_distribution)
        
        for t in range(0, exp.T+1, 1):
            wealth_distribution = np.zeros(exp.max_wealth)
            for wealth in range(exp.max_wealth):
                alpha = ALPHA_HISTORY[n][t][wealth]
                z = Z_HISTORY[n][t][wealth]
            
        
    #             The following is code which factors in probability of win/loss instead of simply expected reward
                wp = win_probability(alpha, ALPHA_BAR_HISTORY[n][t], exp.num_agents)
                wr = win_reward(z, ALPHA_BAR_HISTORY[n][t], exp.beta, exp.k, exp.mining_cost, 
                                exp.num_agents, exp.block_reward) - alpha * exp.mining_cost
                lr = - alpha * exp.mining_cost
                
                win_wealth = round(wealth + wr)
                lose_wealth = round(wealth + lr)
                if lose_wealth < 0:
                    # print(wealth + lr)
                    # print("negative wealth detected")
                    lose_wealth = 0
                wealth_distribution[min(win_wealth, exp.max_wealth) - 1] += WEALTH[t][wealth] * wp
                wealth_distribution[min(lose_wealth, exp.max_wealth) - 1] += WEALTH[t][wealth] * (1 - wp)
                
            WEALTH.append(wealth_distribution)
                
        # save wealth distribution from time T
        WEALTH_HISTORY.append(WEALTH)
            
        ALPHA_BAR = []
        for t in range(0, exp.T+1):
            avg = 0.
            for wealth in range(exp.max_wealth):
                alpha = ALPHA_HISTORY[n][t][wealth]
                density = WEALTH_HISTORY[n][t][wealth]
                avg += alpha * density
            print(avg)
            new_alpha_bar = exp.momentum * ALPHA_BAR_HISTORY[n][t] + (1 - exp.momentum) * avg
            ALPHA_BAR.append(new_alpha_bar)
            
        print(ALPHA_BAR)
        ALPHA_BAR_HISTORY.append(ALPHA_BAR)

    exp.add_results(VALUE_HISTORY, ALPHA_HISTORY, WEALTH_HISTORY, ALPHA_BAR_HISTORY, Z_HISTORY)
    exp.save_to_file(sys.argv[1])



if __name__ == "__main__":
    main()