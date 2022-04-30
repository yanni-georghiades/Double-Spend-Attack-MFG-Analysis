from adversary_model import adv_prob, A
from fee_function import fee


def reward(exp, alpha, z, alpha_bar):
    return R(alpha, z, alpha_bar, exp.beta, exp.k, exp.mining_cost, 
             exp.num_agents, exp.block_reward)

def win_reward(z, alpha_bar, beta, k, mining_cost, num_agents, block_reward):
    return (1-adv_prob(alpha_bar, beta, k)*A(z, alpha_bar, beta, k, block_reward, num_agents, mining_cost)) \
            *(block_reward + fee(z))


def win_probability(alpha, alpha_bar, num_agents):
    return alpha/(alpha + num_agents*alpha_bar)


# the reward function of the honest agents
def R(alpha, z, alpha_bar, beta, k, mining_cost, num_agents, block_reward):
    wr = win_reward(z, alpha_bar, beta, k, mining_cost, num_agents, block_reward)
    wp = win_probability(alpha, alpha_bar, num_agents)
    return wr*wp - alpha*mining_cost