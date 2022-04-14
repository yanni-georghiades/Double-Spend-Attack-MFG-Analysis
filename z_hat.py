from adversary_model import adv_cost, adv_prob, adv_reward, A
from matplotlib import pyplot as plt
import numpy as np



def z_hat(alpha_bar, beta, k, block_reward, num_agents, mining_cost):
	C = adv_cost(alpha_bar, beta, k, block_reward, num_agents, mining_cost)
	P = adv_prob(alpha_bar, beta, k)

	z_hat = (C / P - (k+1)*block_reward) / 1.01 # change this when the fee function changes
	return z_hat



alpha_bar = .3
beta = .4
k = 6
block_reward = 10
num_agents = 10
mining_cost = 1

z = z_hat(alpha_bar, beta, k, block_reward, num_agents, mining_cost)

print(z)

# print(A(z-1, alpha_bar, beta, k, block_reward, num_agents, mining_cost))
# print(A(z, alpha_bar, beta, k, block_reward, num_agents, mining_cost))


# fig, axs = plt.subplots(2, 2,figsize=(10,10))

# alpha_bar = 



# axs[0,0].plot(self.ALPHA_BAR[len(self.ALPHA_BAR)-1])
# axs[0,0].set_title("Alpha Bar")
# axs[0,0].set(xlabel='T', ylabel='Mean Hash Power')

# zs = []

# iterable = np.arange(0.1, 5, .1)

# for b in iterable:
# 	zs.append(z_hat(alpha_bar, beta, k, block_reward, num_agents, b))



# plt.plot(iterable, zs)
# plt.show()