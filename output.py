import pickle
import itertools
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


	def add_results(self, VALUE_HISTORY, ALPHA_HISTORY, WEALTH_HISTORY, ALPHA_BAR_HISTORY, Z_HISTORY, ATTACK_HISTORY):
		self.VALUE_HISTORY = VALUE_HISTORY
		self.ALPHA_HISTORY = ALPHA_HISTORY
		self.WEALTH_HISTORY = WEALTH_HISTORY
		self.ALPHA_BAR_HISTORY = ALPHA_BAR_HISTORY
		self.Z_HISTORY = Z_HISTORY
		self.ATTACK_HISTORY = ATTACK_HISTORY


	def save_to_file(self, file_name):
		try:
			with open(file_name, "wb") as f:
				pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
		except Exception as ex:
			print("Error during pickling object (Possibly unsupported):", ex)


	def plot(self):
		fig, axs = plt.subplots(3, 3,figsize=(15,10))


		axs[0,0].plot(self.ALPHA_BAR_HISTORY[-1])
		axs[0,0].set_title("Alpha Bar Final Iteration")
		axs[0,0].set(xlabel='T', ylabel='Mean Hash Power')

		# axs[0,1].plot(self.Z[len(self.Z)-1])
		# axs[0,1].set_title("Z")
		# axs[0,1].set(xlabel='Wealth', ylabel='Transaction Value in Block')
		color = iter(cm.rainbow(np.linspace(0,1,len(self.ALPHA_BAR_HISTORY))))
		for ab in self.ALPHA_BAR_HISTORY:
			c = next(color)
			axs[0,1].plot(ab, c=c)
		axs[0,1].set_title("Alpha Bar Evolution")
		axs[0,1].set(xlabel="T", ylabel="Alpha Bar")

		abs = []
		for ab in self.ALPHA_BAR_HISTORY:
			abs.append(ab[0])
		axs[0,2].plot(abs)
		axs[0,2].set_title("Alpha bar over time")
		axs[0,2].set(xlabel='n', ylabel='Alpha Bar')

		wealth = self.WEALTH_HISTORY[-1]
		color = iter(cm.rainbow(np.linspace(0, 1, len(wealth))))
		for w in wealth:
			c = next(color)
			axs[1,0].plot(w, c=c)
		axs[1,0].set_title("Wealth Evolution in Final Iteration")
		axs[1,0].set(xlabel='Wealth', ylabel='Proportion of Miners')

		zs = []
		for zw, ww in zip(self.Z_HISTORY[-1], self.WEALTH_HISTORY[-1]):
			avg = 0
			for z, w in zip(zw, ww):
				avg += z*w

			zs.append(avg)
		axs[1,1].plot(zs)
		axs[1,1].set_title("Average Z Selected")
		axs[1,1].set(xlabel='T', ylabel='Transaction Value')

		color = iter(cm.rainbow(np.linspace(0,1,len(self.Z_HISTORY[-1]))))
		for z in self.Z_HISTORY[-1]:
			c = next(color)
			axs[1,2].plot(z)
			# print(z)
		axs[1,2].set_title("Z Evolution in Final Iteration")
		axs[1,2].set(xlabel='Wealth', ylabel='Transaction Value')

		ats = []
		for at in self.ATTACK_HISTORY:
			ats.append(at[0])
		axs[2,2].plot(ats)
		axs[2,2].set_title("Attacks over time")
		axs[2,2].set(xlabel='n', ylabel='Attack Intensity')

		axs[2,0].plot(self.VALUE_HISTORY[-1][1])
		axs[2,0].set_title("Value at t=0, n=N")
		axs[2,0].set(xlabel='wealth', ylabel='value')


		plt.show()









