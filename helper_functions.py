import numpy as np
from matplotlib import pyplot as plt

from environment_model import R, win_reward, win_probability, reward
from adversary_model import adv_cost, adv_prob, A
from fee_function import fee

alpha_step_size = 0.1

# def select_best_parameters(exp):
#     z = 0
#     alpha = 0

#     for i in range(5):
#         z = select_best_z(alpha, exp.max_tx_value, exp.alpha_bar_init, exp.beta, exp.k, exp.mining_cost, 
#             exp.num_agents, exp.block_reward)
#         alpha = select_best_alpha(exp.max_wealth, z, exp.alpha_bar_init, exp.beta, exp.k, exp.mining_cost, 
#             exp.num_agents, exp.block_reward)
        
#     return (alpha, z)


# def select_best_z(alpha, max_z, alpha_bar, beta, k, mining_cost, num_agents, block_reward):
#     rewards = []
#     for i in range(max_z + 1):
#         rewards.append(R(alpha, i, alpha_bar, beta, k, mining_cost, num_agents, block_reward))

#     return rewards.index(max(rewards))


# def select_best_alpha(max_alpha, z, alpha_bar, beta, k, mining_cost, num_agents, block_reward):
#     rewards = []

#     for i in np.arange(0, max_alpha + alpha_step_size/2, alpha_step_size):
#         rewards.append(R(i, z, alpha_bar, beta, k, mining_cost, num_agents, block_reward))

#     return rewards.index(max(rewards)) * alpha_step_size


# def best_actions(exp, alpha_bar, wealth, values):
#     C = adv_cost(alpha_bar, exp.beta, exp.k, exp.block_reward, 
#                      exp.num_agents, exp.mining_cost)
#     P = adv_prob(alpha_bar, exp.beta, exp.k)

#     zh = (C / P - (exp.k+1)*exp.block_reward) / 1.01 # change this when the fee function changes

#     if zh < 0:
#         zh = exp.max_tx_value
#     # we know that one of z_hat or max_tx_value will always be the best choice
#     z1 = zh
#     z2 = exp.max_tx_value
    

#     max_alpha = wealth / exp.mining_cost

#     # test z_hat
#     rewards1 = []
#     for alpha_i in np.arange(0, max_alpha + alpha_step_size/2, alpha_step_size):
#         wr = win_reward(z1, alpha_bar, exp.beta, exp.k, exp.mining_cost, 
#                         exp.num_agents, exp.block_reward) - alpha_i * exp.mining_cost
#         wp = win_probability(alpha_i, alpha_bar, exp.num_agents)
#         lr = - alpha_i * exp.mining_cost
        
#         win_wealth = round(wealth + wr)
#         win_value = values[min(win_wealth, exp.max_wealth) - 1]
#         lose_wealth = round(wealth + lr)
#         if lose_wealth < 0:
#             print("wealth is negative")
#         lose_value = values[min(lose_wealth, exp.max_wealth) - 1]
#         rew = reward(exp, alpha_i, z1, alpha_bar)
        
#         rewards1.append(rew + wp * win_value + (1 - wp) * lose_value)
        
#     # test max z
#     rewards2 = []
#     for alpha_i in np.arange(0, max_alpha + alpha_step_size/2, alpha_step_size):
#         wr = win_reward(z2, alpha_bar, exp.beta, exp.k, exp.mining_cost, 
#                         exp.num_agents, exp.block_reward) - alpha_i * exp.mining_cost
#         wp = win_probability(alpha_i, alpha_bar, exp.num_agents)
#         lr = - alpha_i * exp.mining_cost
        
#         win_wealth = round(wealth + wr)
#         win_value = values[min(win_wealth, exp.max_wealth) - 1]
#         lose_wealth = round(wealth + lr)
#         if lose_wealth < 0:
#             print("wealth is negative")
#         lose_value = values[min(lose_wealth, exp.max_wealth) - 1]
#         rew = reward(exp, alpha_i, z2, alpha_bar)
        
#         rewards2.append(rew + wp * win_value + (1 - wp) * lose_value)

#     # print("rewards1")
#     # print(rewards1)
#     # print("rewards2")
#     # print(rewards2)

    
#     # select highest reward. in case of a tie, go with z_hat
#     if max(rewards1) >= max(rewards2) and zh < exp.max_tx_value:
#         z = z1
#         alpha = rewards1.index(max(rewards1)) * alpha_step_size
#         val = max(rewards1)
#     else:
#         z = z2
#         alpha = rewards2.index(max(rewards2)) * alpha_step_size
#         val = max(rewards2)
    
#     P = adv_prob(alpha_bar, exp.beta, exp.k)
#     if z > zh:
#         analytical = np.sqrt((1 - P)*(exp.block_reward + fee(z)) * exp.num_agents * alpha_bar / exp.mining_cost) - exp.num_agents * alpha_bar
#     else:
#         analytical = np.sqrt((exp.block_reward + fee(z)) * exp.num_agents * alpha_bar / exp.mining_cost) - exp.num_agents * alpha_bar

#     print("alpha chosen: " + str(alpha))
#     print("analytical alpha: " + str(analytical))
#     # print(alpha)
#     # print(z)
#     # print(val)

#     return (alpha, z, val)


def best_actions2(exp, alpha_bar, wealth, values):
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


def converged(history):
    check = True
    ab1 = history[-1]
    ab2 = history[-2]
    for i in range(len(ab1)):
        if abs(ab1[i] - ab2[i]) > .001:
            check = False
    return check
