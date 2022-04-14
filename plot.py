import pickle
import sys

from output import Experiment

filename = sys.argv[1]

try:
    with open(filename, "rb") as f:
        experiment = pickle.load(f)
except Exception as ex:
    print("Error during unpickling object (Possibly unsupported):", ex)

print(experiment.Z)

experiment.plot()