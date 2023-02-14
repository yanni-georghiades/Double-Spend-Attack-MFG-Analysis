import pickle
import sys

from output import Experiment

filename = sys.argv[1]

try:
    with open(filename, "rb") as f:
        experiment = pickle.load(f)
except Exception as ex:
    print("Error during unpickling object (Possibly unsupported):", ex)



print("Experiment Information:")
print("=" * 20)
print("k: " + str(experiment.k))
print("beta: " + str(experiment.beta))
print("block reward: " + str(experiment.block_reward))
print("alpha bar init: " + str(experiment.alpha_bar_init))
print("mining cost: " + str(experiment.mining_cost))
print("number of agents: " + str(experiment.num_agents))
print("max wealth: " + str(experiment.max_wealth))
print("max z: " + str(experiment.max_tx_value))
print("T: " + str(experiment.T))
print("N: " + str(experiment.N))
print("momentum: " + str(experiment.momentum))

# print(experiment.ATTACK_HISTORY)

last_wealths = experiment.WEALTH_HISTORY[-1][-1]
avg = 0
for i in range(len(last_wealths)):
	avg += i * last_wealths[i]
print("Average ending wealth: " + str(avg))

experiment.plot()