import numpy as np
import torch as tc
import random

SEED = 42

np.random.seed(SEED)
tc.manual_seed(SEED) 
random.seed(SEED)
################## Definindo os parametros #######################

Js          = [random.uniform(-1,1) for _ in range(15)]
dissipation = [random.uniform(0,1) for _ in range(4)]

print(f"Js: {Js}")
print(f"dissipation: {dissipation}")