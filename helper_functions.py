import numpy as np

from environment_model import R, win_reward, win_probability, reward, beta_default, block_reward_default, \
                                mining_cost_default, num_agents_default, \
                                k_default, max_wealth, max_tx_value
from adversary_model import adv_cost, adv_prob

alpha_step_size = 0.1



def select_best_parameters(init_alpha, max_alpha, init_z, max_z, iterations, alpha_bar, beta, 
                           k, mining_cost, num_agents, block_reward, print_summary=True):
    z = init_z
    alpha = init_alpha
    
    z_history = []
    alpha_history = []
    for i in range(iterations):
        z_history.append(z)
        alpha_history.append(alpha)
        
        z = select_best_z(alpha, max_z, alpha_bar, beta, k, mining_cost, num_agents, block_reward)
        alpha = select_best_alpha(max_alpha, z, alpha_bar, beta, k, mining_cost, num_agents, block_reward)
        
    if print_summary:
        parameter_summary(z, alpha, z_history, alpha_history, max_z, max_alpha, alpha_bar, 
                          beta, k, mining_cost, num_agents, block_reward)
        
    return (alpha, z)


def select_best_z(alpha, max_z, alpha_bar, beta, k, mining_cost, num_agents, block_reward, plot=False):
    rewards = []
    for i in range(max_z + 1):
        rewards.append(R(alpha, i, alpha_bar, beta, k, mining_cost, num_agents, block_reward))
        
    if plot==True:
        plt.plot(rewards, 'ro')
    return rewards.index(max(rewards))


def select_best_alpha(max_alpha, z, alpha_bar, beta, k, mining_cost, num_agents, block_reward, plot=False):
    rewards = []
    for i in np.arange(0, max_alpha + alpha_step_size/2, alpha_step_size):
        rewards.append(R(i, z, alpha_bar, beta, k, mining_cost, num_agents, block_reward))

    if plot==True:
        plt.plot(rewards, 'ro')

    return rewards.index(max(rewards)) * alpha_step_size


def next_wealth(wealth, alpha, z, alpha_bar):
    return round(wealth + reward(alpha, z, alpha_bar))


def best_actions(alpha_bar, wealth, values):
    C = adv_cost(alpha_bar, beta_default, k_default, block_reward_default, 
                     num_agents_default, mining_cost_default)
    P = adv_prob(alpha_bar, beta_default, k_default)

    z_hat = (C / P - (k_default+1)*block_reward_default) / 1.01 # change this when the fee function changes

    # we know that one of z_hat or max_tx_value will always be the best choice
    z1 = z_hat
    z2 = max_tx_value
    
    # test z_hat
    rewards1 = []
    for alpha_i in np.arange(0, wealth + alpha_step_size/2, alpha_step_size):
        wr = win_reward(z1, alpha_bar, beta_default, k_default, mining_cost_default, 
                        num_agents_default, block_reward_default) - alpha_i * mining_cost_default
        wp = win_probability(alpha_i, alpha_bar, num_agents_default)
        lr = - alpha_i * mining_cost_default
        
        win_wealth = round(wealth + wr)
        win_value = values[min(win_wealth, max_wealth) - 1]
        lose_wealth = round(wealth + lr)
        if lose_wealth < 0:
            print("wealth is negative")
        lose_value = values[min(lose_wealth, max_wealth) - 1]
        rew = reward(alpha_i, z1, alpha_bar)
        
        rewards1.append(rew + wp * win_value + (1 - wp) * lose_value)
        
        
#         new_wealth = next_wealth(wealth, alpha_i, z1, alpha_bar)
#         rewards1.append(rew + values[min(new_wealth, max_wealth) - 1])
        
    # test max z
    rewards2 = []
    for alpha_i in np.arange(0, wealth + alpha_step_size/2, alpha_step_size):
        wr = win_reward(z2, alpha_bar, beta_default, k_default, mining_cost_default, 
                        num_agents_default, block_reward_default) - alpha_i * mining_cost_default
        wp = win_probability(alpha_i, alpha_bar, num_agents_default)
        lr = - alpha_i * mining_cost_default
        
        win_wealth = round(wealth + wr)
        win_value = values[min(win_wealth, max_wealth) - 1]
        lose_wealth = round(wealth + lr)
        if lose_wealth < 0:
            print("wealth is negative")
        lose_value = values[min(lose_wealth, max_wealth) - 1]
        rew = reward(alpha_i, z2, alpha_bar)
        
        rewards2.append(rew + wp * win_value + (1 - wp) * lose_value)
        
        
#         new_wealth = next_wealth(wealth, alpha_i, z1, alpha_bar)
#         rewards2.append(rew + values[min(new_wealth, max_wealth) - 1])
        
        
        
#         rew = reward(alpha_i, z2, alpha_bar)
#         new_wealth = next_wealth(wealth, alpha_i, z2, alpha_bar)
#         rewards2.append(rew + values[min(new_wealth, max_wealth) - 1])
    
    # select highest reward. in case of a tie, go with z_hat
    if max(rewards1) >= max(rewards2) and z_hat < max_tx_value:
        z = z1
        alpha = rewards1.index(max(rewards1)) * alpha_step_size
        val = max(rewards1)
    else:
        z = z2
        alpha = rewards2.index(max(rewards2)) * alpha_step_size
        val = max(rewards2)
    
    return (alpha, z, val)
