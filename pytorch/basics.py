# These notes are on the PyTorch library. Pytorch and Numpy works same.

import torch
import numpy as np

# the only difference between PyTorch and nump is that
# Numpy is only processed over cpu which is slow. Pytorch can process it
# on GPU as well but we need to explicitly tell it to use GPU.
# this checks if we have GPUs available or not. if returned true, means
# we have NVIDIA's GPUs.
print("is gpu available?: ",torch.cuda.is_available())

# the resulr of these print statements will show that both 
# are exactly same.
print("Numpy Array:", np.array([[1,2,3],[4,5,6]]))
print("PyTorch Tensor:", torch.tensor([[1,2,3],[4,5,6]]))

print ("Numpy Ones: ", np.ones((2,4)))
print("Torch ones: ", torch.ones((2,4)))

print ("Numpy Zeros: ", np.zeros((2,4)))
print("Torch Zeros: ", torch.zeros((2,4)))

print ("Numpy random: ", np.random.random((2,4)))
print("Torch random: ", torch.rand((2,4)))

