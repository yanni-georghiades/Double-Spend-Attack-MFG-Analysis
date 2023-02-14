import numpy as np
from matplotlib import pyplot as plt

from environment_model import R, win_reward, win_probability, reward
from adversary_model import adv_cost, adv_prob, A
from fee_function import fee

alpha_step_size = 0.1


def best_actions(exp, alpha_bar, wealth, values):
    P = adv_prob(alpha_bar, exp.beta, exp.k)

    N = 100000
    z = np.linspace(0,exp.max_tx_value,N,endpoint=True)

    T = A(z, alpha_bar, exp.beta, exp.k, exp.block_reward, exp.num_agents, exp.mining_cost)

    max_alpha = wealth / exp.mining_cost
    # alpha = np.sqrt((1 - P*T)*(exp.block_reward + fee(z)) * exp.num_agents * alpha_bar / exp.mining_cost) \
    #                 - exp.num_agents * alpha_bar
    alpha = np.sqrt((exp.block_reward + fee(z)) * exp.num_agents * alpha_bar / exp.mining_cost) \
                    - exp.num_agents * alpha_bar
    alpha = np.minimum(alpha, max_alpha * np.ones(z.shape))
    alpha = np.maximum(alpha, np.zeros(z.shape))
    
    wr = exp.block_reward + fee(z) - alpha * exp.mining_cost
    wp = alpha / (alpha + exp.num_agents * alpha_bar)
    lr = - alpha * exp.mining_cost
    
    win_wealth = np.round(wealth + wr)
    vals = np.array(values,)
    win_value = vals[np.minimum(win_wealth, exp.max_wealth).astype(int) - 1]
    lose_wealth = np.round(wealth + lr)
    lose_value = vals[np.maximum(lose_wealth, 1).astype(int) - 1]

    # F = (1 - P * T) * (exp.block_reward + fee(z)) * wp  + lr + wp * win_value + (1 - wp) * lose_value
    F = (exp.block_reward + fee(z)) * wp  + lr + wp * win_value + (1 - wp) * lose_value
    idx = np.argmax(F)
    val = F[idx]
    z_ret = z[idx]

    # if T[idx] > .5 and wealth >= 20:
    #     print(alpha[idx])
    #     print(z_ret)
    #     print(val)
    #     # for i in range(z.shape[0]):
    #     #     print('=' * 10)
    #     #     print(F[i])
    #     #     print(z[i])
    #     #     print(alpha[i])

    #     C = adv_cost(alpha_bar, exp.beta, exp.k, exp.block_reward, 
    #                  exp.num_agents, exp.mining_cost)
    #     P = adv_prob(alpha_bar, exp.beta, exp.k)

    #     zh = (C / P - (exp.k+1)*exp.block_reward) / 1.01 # change this when the fee function changes
    #     print(zh)

    #     zh_idx = np.where(zh)[0]
    #     print(alpha[zh_idx])
    #     print(F[zh_idx])

    #     print(wealth)
    #     plt.figure()
    #     plt.plot(alpha)
    #     plt.figure()
    #     plt.plot(F)
    #     plt.figure()
    #     plt.plot(lose_wealth)
    #     plt.figure()
    #     plt.plot(win_wealth)
    #     plt.figure()
    #     plt.plot(vals)
    #     plt.show()

    return (alpha[idx], z_ret, val)
    # return (0,0,0)


def converged(history):
    check = True
    ab1 = history[-1]
    ab2 = history[-2]
    for i in range(len(ab1)):
        if abs(ab1[i] - ab2[i]) > .001:
            check = False
    return check
