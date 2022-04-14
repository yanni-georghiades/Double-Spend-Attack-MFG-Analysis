from adversary_model import adv_prob, A
from fee_function import fee

# default parameters
k_default = 6
beta_default = .4
block_reward_default = 10
mining_cost_default = 1. 
num_agents_default = 10

max_wealth = 100
max_tx_value = 100



def reward(alpha, z, alpha_bar):
    return R(alpha, z, alpha_bar, beta_default, k_default, mining_cost_default, 
             num_agents_default, block_reward_default)

def win_reward(z, alpha_bar, beta, k, mining_cost, num_agents, block_reward):
    return (1-(1-adv_prob(alpha_bar, beta, k))*A(z, alpha_bar, beta, k, block_reward, num_agents, mining_cost)) \
            *(block_reward + fee(z))


def win_probability(alpha, alpha_bar, num_agents):
    return alpha/(alpha + num_agents*alpha_bar)


# the reward function of the honest agents
def R(alpha, z, alpha_bar, beta, k, mining_cost, num_agents, block_reward):
    wr = win_reward(z, alpha_bar, beta, k, mining_cost, num_agents, block_reward)
    wp = win_probability(alpha, alpha_bar, num_agents)
    return wr*wp - alpha*mining_cost