    # <2020.03.30 (Mon)>
    # Linear Regression: H(x) & cost(W, or(2.1471, grad_fn=<MseLossBackward>)

    # (1)
    # 0. Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# For reproducibility
torch.manual_seed(1)


    # 1. Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

print(x_train)
print(x_train.shape)
# tensor([[1.],
#         [2.],
#         [3.]])
# torch.Size([3, 1])

print(y_train)
print(y_train.shape)
# tensor([[1.],
#         [2.],
#         [3.]])
# torch.Size([3, 1])


    # 2. Weight Initialization: weight & bias
W = torch.zeros(1, requires_grad=True)      #initialize & specify that they will learned
print(W)
# tensor([0.], requires_grad=True)

b = torch.zeros(1, requires_grad=True)      #initialize & specify that they will learned
print(b)
# tensor([0.], requires_grad=True)


    # 3. Hypothesis: H(x) = Wx + b
hypothesis = x_train * W + b
print(hypothesis)
# tensor([[0.],
#         [0.],
#         [0.]], grad_fn=<AddBackward0>)


    # 4. Cost: cost(W, b) = mean{(hypothesis - y_train)^2}
print(hypothesis)
# tensor([[0.],
#         [0.],
#         [0.]], grad_fn=<AddBackward0>)
print(y_train)
# tensor([[1.],
#         [2.],
#         [3.]])
print(hypothesis - y_train)
# tensor([[-1.],
#         [-2.],
#         [-3.]], grad_fn=<SubBackward0>)
print((hypothesis - y_train)**2)
# tensor([[1.],
#         [4.],
#         [9.]], grad_fn=<PowBackward0>)

cost = torch.mean((hypothesis - y_train)**2)
print(cost) 
# tensor(4.6667, grad_fn=<MeanBackward0>)


    ## 5. Gradient Descent
optimizer = optim.SGD([W, b], lr = 0.01)        #SGD: stochastic gradient descent, [W,b]: tensors to learn, lr: learning rate

optimizer.zero_grad()       # initialize gradient
cost.backward()             # compute gradiend
optimizer.step()            # improvement in direction

print(W)
# tensor([0.0933], requires_grad=True)
print(b)
# tensor([0.0400], requires_grad=True)

#Let's check if the hypothesis is now better
hypothesis = x_train * W + b
print(hypothesis)
# tensor([[0.1333],
#         [0.2267],
#         [0.3200]], grad_fn=<AddBackward0>)

cost = torch.mean((hypothesis - y_train)**2)
print(cost)
# tensor(3.6927, grad_fn=<MeanBackward0>)



    ## 6. ==> Training with Full Code -----------------------------------------------------
# Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

# initialize model
W = torch.zeros(1, requires_grad=True)    
b = torch.zeros(1, requires_grad=True)    

# optimizer settings
optimizer = optim.SGD([W, b], lr = 0.01)  

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
    # compute H(x)
    hypothesis = x_train * W + b

    # compute cost
    cost = torch.mean((hypothesis - y_train)**2)

    # improve h(x) at cost
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # print the log every 100 times
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()
        ))
    
# Epoch    0/1000 W: 0.093, b: 0.040 Cost: 4.666667
# Epoch  100/1000 W: 0.873, b: 0.289 Cost: 0.012043
# Epoch  200/1000 W: 0.900, b: 0.227 Cost: 0.007442
# Epoch  300/1000 W: 0.921, b: 0.179 Cost: 0.004598
# Epoch  400/1000 W: 0.938, b: 0.140 Cost: 0.002842
# Epoch  500/1000 W: 0.951, b: 0.110 Cost: 0.001756
# Epoch  600/1000 W: 0.962, b: 0.087 Cost: 0.001085
# Epoch  700/1000 W: 0.970, b: 0.068 Cost: 0.000670
# Epoch  800/1000 W: 0.976, b: 0.054 Cost: 0.000414
# Epoch  900/1000 W: 0.981, b: 0.042 Cost: 0.000256
# Epoch 1000/1000 W: 0.985, b: 0.033 Cost: 0.000158
    ## --------------------------------------------------------------------------------------

print()
print()
    ## (2) High-level implementation with 'nn.Module'
    ## 1. Linear Regression Model
# Remember that we had this fake data.
x_train = torch.FloatTensor([[1], [2], [3]])        
y_train = torch.FloatTensor([[1], [2], [3]])      

# Basically, all models of PyTorch are created by inheriting the provided 'nn.Module'
class LinearRegressionModel(nn.Module):
    # __init__: define the layer to use
    # nn.linear: linear regression model
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    # forward: tells how to calculate the output value from the input value
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

    
    ## 2. Hypothesis: H(x)
hypothesis = model(x_train)
print(hypothesis)
# tensor([[0.0739],
#         [0.5891],
#         [1.1044]], grad_fn=<AddmmBackward>)

        
    ## 3. Cost: MSE(mean squared error)
print(hypothesis)
# tensor([[0.0739],
#         [0.5891],
#         [1.1044]], grad_fn=<AddmmBackward>)
print(y_train)
# tensor([[1.],
#         [2.],
#         [3.]])

cost = F.mse_loss(hypothesis, y_train)
print(cost)
# tensor(2.1471, grad_fn=<MseLossBackward>)


    ## 4. Gradient Descent
optimizer = optim.SGD(model.parameters(), lr = 0.01)

optimizer.zero_grad()
cost.backward()
optimizer.step()

     

    ## 5. ==> Training with Full Code -----------------------------------------------------
# Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

# initialize model
    # W = torch.zeros(1, requires_grad=True)    
    # b = torch.zeros(1, requires_grad=True)    
model = LinearRegressionModel()

# optimizer settings
    # optimizer = optim.SGD([W, b], lr = 0.01)  
optimizer = optim.SGD(model.parameters(), lr = 0.01)  

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
    # compute H(x)
        # hypothesis = x_train * W + b
    prediction = model(x_train)

    # compute cost
        # cost = torch.mean((hypothesis - y_train)**2)
        # cost = F.mse_loss(hypothesis, y_train)
    cost = F.mse_loss(prediction, y_train)

    # improve h(x) at cost
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # print the log every 100 times
    if epoch % 100 == 0:
        params = list(model.parameters())
        W = params[0].item()
        b = params[1].item()
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W, b, cost.item()
        ))
    
# Epoch    0/1000 W: -0.101, b: 0.508 Cost: 4.630286
# Epoch  100/1000 W: 0.713, b: 0.653 Cost: 0.061555
# Epoch  200/1000 W: 0.774, b: 0.514 Cost: 0.038037
# Epoch  300/1000 W: 0.822, b: 0.404 Cost: 0.023505
# Epoch  400/1000 W: 0.860, b: 0.317 Cost: 0.014525
# Epoch  500/1000 W: 0.890, b: 0.250 Cost: 0.008975
# Epoch  600/1000 W: 0.914, b: 0.196 Cost: 0.005546
# Epoch  700/1000 W: 0.932, b: 0.154 Cost: 0.003427
# Epoch  800/1000 W: 0.947, b: 0.121 Cost: 0.002118
# Epoch  900/1000 W: 0.958, b: 0.095 Cost: 0.001309
# Epoch 1000/1000 W: 0.967, b: 0.075 Cost: 0.000809
    
    ## --------------------------------------------------------------------------------------



