    # <2020.04.03 (Fri)>
    # Deeper Look at GD (Gradient Descent)

    ## 0. imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#For reproducibility
torch.manual_seed(1)

   
    ## 1. Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

plt.scatter(x_train, y_train)

#Best-fit line
xs = np.linspace(1, 3, 1000)
plt.plot(xs, xs)
#plt.show()
    

    ## 2. Cost by W: H(x) = W*x
W_l = np.linspace(-5, 7, 1000)
cost_l = []
for W in W_l:
    hypothesis = W * x_train
    cost = torch.mean((hypothesis - y_train) ** 2)

    cost_l.append(cost.item())

plt.plot(W_l, cost_l)
plt.xlabel('$W$')
plt.ylabel('Cost')
#plt.show()
    

    ## 3. Gradient Descent by Hand
W = 0

gradient = torch.sum((W * x_train - y_train) * x_train)     # gradient W
print(gradient)

lr = 0.1                # learning rate
W -= lr * gradient      # W:= W - lr* GD(W)
print(W)


    ## 4-1 Training -----------------------------------------------------------------------
# data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

# model initiallize
W = torch.zeros(1)

# setting learning rate 
lr = 0.1

nb_epochs = 10
for epoch in range(nb_epochs + 1):
    
    # compute 'H(x)'
    hypothesis = x_train * W

    # compute 'cost gradient'
    cost = torch.mean((hypothesis - y_train) ** 2)
    gradient = torch.sum((W * x_train - y_train) * x_train)

    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(
        epoch, nb_epochs, W.item(), cost.item()
        ))

    # improve H(x) using cost gradient ==> Gradient Descent
    W -= lr * gradient

#Epoch    0/10 W: 0.000, Cost: 4.666667
#Epoch    1/10 W: 1.400, Cost: 0.746666
#Epoch    2/10 W: 0.840, Cost: 0.119467
#Epoch    3/10 W: 1.064, Cost: 0.019115
#Epoch    4/10 W: 0.974, Cost: 0.003058
#Epoch    5/10 W: 1.010, Cost: 0.000489
#Epoch    6/10 W: 0.996, Cost: 0.000078
#Epoch    7/10 W: 1.002, Cost: 0.000013
#Epoch    8/10 W: 0.999, Cost: 0.000002
#Epoch    9/10 W: 1.000, Cost: 0.000000
#Epoch   10/10 W: 1.000, Cost: 0.000000


    ## 4-2 Training with 'optim' ----------------------------------------------------------
# data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

# model initiallize
W = torch.zeros(1, requires_grad=True)          # initiallize & specify that they will learned

# setting optimizer
optimizer = optim.SGD([W], lr = 0.15)           # SGD: stochastic gradient descent, [W]: learning model, lr: learning rate

nb_epochs = 10
for epoch in range(nb_epochs + 1):
    
    # compute 'H(x)'
    hypothesis = x_train * W

    # compute 'cost gradient'
    cost = torch.mean((hypothesis - y_train) ** 2)

    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(
        epoch, nb_epochs, W.item(), cost.item()
        ))

    # improve H(x) using cost ==> Gradient Descent
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

#Epoch    0/10 W: 0.000, Cost: 4.666667
#Epoch    1/10 W: 1.400, Cost: 0.746666
#Epoch    2/10 W: 0.840, Cost: 0.119467
#Epoch    3/10 W: 1.064, Cost: 0.019115
#Epoch    4/10 W: 0.974, Cost: 0.003058
#Epoch    5/10 W: 1.010, Cost: 0.000489
#Epoch    6/10 W: 0.996, Cost: 0.000078
#Epoch    7/10 W: 1.002, Cost: 0.000013
#Epoch    8/10 W: 0.999, Cost: 0.000002
#Epoch    9/10 W: 1.000, Cost: 0.000000
#Epoch   10/10 W: 1.000, Cost: 0.000000
