import pickle
import sys

from output import Experiment

filename = sys.argv[1]

try:
    with open(filename, "rb") as f:
        experiment = pickle.load(f)
except Exception as ex:
    print("Error during unpickling object (Possibly unsupported):", ex)


# for n in range(experiment.N):
# 	print("iteration: " + str(n) + "\n\n")
# 	print("value")
# 	print(experiment.VALUE_HISTORY[n])
# 	print("alpha")
# 	print(experiment.ALPHA_HISTORY[n])
# 	print("wealth")
# 	print(experiment.WEALTH_HISTORY[n])
# 	print("alpha bar")
# 	print(experiment.ALPHA_BAR_HISTORY[n])
# 	print("z")
# 	print(experiment.Z_HISTORY[n])


experiment.plot()