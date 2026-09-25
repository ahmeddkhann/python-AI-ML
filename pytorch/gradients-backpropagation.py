# the pytorch library helps us to calculate gradients and do back
# propagation easily. it is helpul as it does all the process
# automatically.

import torch

# this line creates a tensor.
# requires_grad tells pytorch that keep track of the operations on
# a as i might want derivateive later with respect to a.
a = torch.tensor(3.0, requires_grad=True)

# we are perfoming some mathemathical operations on a.
# b = 4a + 2
b = a * 4 + 2 

# this line will result in outputing 
# tensor(14., grad_fn=<AddBackward0>)
# 14 is the mathemathical value that came out of the operation while the
# grad_fn=<AddBackward0> tells that this tensor was produced due to an
# addition operation and it can backpropagate how this tensor was created.
print (b)

# this line is doing backward pass.
b.backward()

# printing gradient of the final result b with respect to a.
print(a.grad)


# lets do an operation c on the tensor a.
# c = a5 
c = a ** 5

# this will result in tensor(243., grad_fn=<PowBackward0>)
# where 243 is the result to the mathemathical operation while
# grad_fn=<PowBackward0> tells us that this tensor was produced due
# to a power operation and it can back propagate how this tensor was
# created.
print(c)
# backward pass.
c.backward()
# printing gradient.
print(a.grad)


# working with multiple tensors and one operation.
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)
z = 2 * x + 3 * y
print(z)
z.backward()
print("Gradient of x:", x.grad)
print("Gradient of y:", y.grad)


# multiple tensors and multiple operations on it.
tensor1 = torch.tensor(3.0, requires_grad=True)
tensor2 = torch.tensor(4.0, requires_grad=True)
operation1 = tensor1 * tensor2
operation2 = operation1 + tensor1
operation3 = operation2 ** tensor2
print("operation1 =", operation1)
print("operation2 =", operation2)
print("opeation3 =", operation3)
operation3.backward()
print("tensor1.grad =", tensor1.grad)
print("tensor2.grad =", tensor2.grad)