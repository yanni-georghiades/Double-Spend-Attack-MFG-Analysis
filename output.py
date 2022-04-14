import pickle
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

class Experiment():
	def __init__(self, k, beta, block_reward, alpha_bar_init, mining_cost, num_agents, max_wealth, max_tx_value,
					T, N, momentum):
		self.k = k
		self.beta = beta
		self.block_reward = block_reward
		self.alpha_bar_init = alpha_bar_init
		self.mining_cost = mining_cost
		self.num_agents = num_agents

		self.max_wealth = max_wealth
		self.max_tx_value = max_tx_value

		self.T = T
		self.N = N
		self.momentum = momentum


	def add_results(self, VALUE_HISTORY, ALPHA, WEALTH_HISTORY, ALPHA_BAR, Z):
		self.VALUE_HISTORY = VALUE_HISTORY
		self.ALPHA = ALPHA
		self.WEALTH_HISTORY = WEALTH_HISTORY
		self.ALPHA_BAR = ALPHA_BAR
		self.Z = Z


	def save_to_file(self, file_name):
		try:
			with open(file_name, "wb") as f:
				pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
		except Exception as ex:
			print("Error during pickling object (Possibly unsupported):", ex)


	def plot(self):
		fig, axs = plt.subplots(2, 2,figsize=(10,10))


		axs[0,0].plot(self.ALPHA_BAR[len(self.ALPHA_BAR)-1])
		axs[0,0].set_title("Alpha Bar")
		axs[0,0].set(xlabel='T', ylabel='Mean Hash Power')

		axs[0,1].plot(self.Z[len(self.Z)-1])
		axs[0,1].set_title("Z")
		axs[0,1].set(xlabel='Wealth', ylabel='Transaction Value in Block')

		wealth = self.WEALTH_HISTORY[len(self.WEALTH_HISTORY)-1]
		color = iter(cm.rainbow(np.linspace(0, 1, len(wealth))))
		for w in wealth:
			c = next(color)
			axs[1,0].plot(w, c=c)
		axs[1,0].set_title("Wealth Evolution in Final Iteration")
		axs[1,0].set(xlabel='Wealth', ylabel='Proportion of Miners')

		plt.show()








