    # <2020.04.03 (Fri)>
    # Multivariate Linear Regression

    ## 0. imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#For reproducibility
torch.manual_seed(1)


    ## 1. Naive Data Representation: H(x1, x2, x3) = x1w1 + x2w2 + x3w3 + b
#data
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])    
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])    
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])    
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]]) 

#model initialization
w1 = torch.zeros(1, requires_grad = True)   
w2 = torch.zeros(1, requires_grad = True)   
w3 = torch.zeros(1, requires_grad = True)   
b = torch.zeros(1, requires_grad = True)   

#setting optimizer
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    #compute H(x)
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    #compute cost
    cost = torch.mean((hypothesis - y_train) ** 2)

    #improve H(x) using cost
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    #print log every 100 times
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
            ))

#Epoch    0/1000 w1: 0.294 w2: 0.294 w3: 0.297 b: 0.003420 Cost: 29661.800781
#Epoch  100/1000 w1: 0.674 w2: 0.661 w3: 0.676 b: 0.007920 Cost: 1.563628
#Epoch  200/1000 w1: 0.679 w2: 0.655 w3: 0.677 b: 0.008070 Cost: 1.497595
#Epoch  300/1000 w1: 0.684 w2: 0.649 w3: 0.677 b: 0.008219 Cost: 1.435044
#Epoch  400/1000 w1: 0.689 w2: 0.643 w3: 0.678 b: 0.008367 Cost: 1.375726
#Epoch  500/1000 w1: 0.694 w2: 0.638 w3: 0.678 b: 0.008514 Cost: 1.319497
#Epoch  600/1000 w1: 0.699 w2: 0.633 w3: 0.679 b: 0.008659 Cost: 1.266215
#Epoch  700/1000 w1: 0.704 w2: 0.627 w3: 0.679 b: 0.008804 Cost: 1.215703
#Epoch  800/1000 w1: 0.709 w2: 0.622 w3: 0.679 b: 0.008948 Cost: 1.167810
#Epoch  900/1000 w1: 0.713 w2: 0.617 w3: 0.680 b: 0.009090 Cost: 1.122429
#Epoch 1000/1000 w1: 0.718 w2: 0.613 w3: 0.680 b: 0.009232 Cost: 1.079390
        

    ## 2. Matrix Data Representation: H(x) = XW
#data
# x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])    
# x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])    
# x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])    
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])    
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]]) 

print(x_train.shape)
# torch.Size([5, 3])
print(y_train.shape)
# torch.Size([5, 1])

#model initialization
# w1 = torch.zeros(1, requires_grad = True)   
# w2 = torch.zeros(1, requires_grad = True)   
# w3 = torch.zeros(1, requires_grad = True)   
W = torch.zeros((3, 1), requires_grad = True)   
b = torch.zeros(1, requires_grad = True)   

#setting optimizer
# optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

    #compute H(x)
    # hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b
    hypothesis = x_train.matmul(W) + b      # or .mm or @

    #compute cost
    cost = torch.mean((hypothesis - y_train) ** 2)

    #improve H(x) using cost
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    #print log every 100 times
    # if epoch % 100 == 0:
        # print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:3f} Cost: {:.6f}'.format(
        #     epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        #     ))
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
        ))
# Epoch    0/20 hypothesis: tensor([0., 0., 0., 0., 0.]) Cost: 29661.800781
# Epoch    1/20 hypothesis: tensor([67.2578, 80.8397, 79.6523, 86.7394, 61.6605]) Cost: 9298.520508
# Epoch    2/20 hypothesis: tensor([104.9128, 126.0990, 124.2466, 135.3015,  96.1821]) Cost: 2915.712402
# Epoch    3/20 hypothesis: tensor([125.9942, 151.4381, 149.2133, 162.4896, 115.5097]) Cost: 915.040649
# Epoch    4/20 hypothesis: tensor([137.7968, 165.6247, 163.1912, 177.7112, 126.3307]) Cost: 287.935822
# Epoch    5/20 hypothesis: tensor([144.4044, 173.5674, 171.0168, 186.2331, 132.3891]) Cost: 91.371170
# Epoch    6/20 hypothesis: tensor([148.1035, 178.0143, 175.3980, 191.0042, 135.7812]) Cost: 29.758301
# Epoch    7/20 hypothesis: tensor([150.1744, 180.5042, 177.8508, 193.6753, 137.6805]) Cost: 10.445318
# Epoch    8/20 hypothesis: tensor([151.3336, 181.8983, 179.2240, 195.1707, 138.7440]) Cost: 4.391228
# Epoch    9/20 hypothesis: tensor([151.9824, 182.6789, 179.9928, 196.0079, 139.3396]) Cost: 2.493135
# Epoch   10/20 hypothesis: tensor([152.3454, 183.1161, 180.4231, 196.4765, 139.6732]) Cost: 1.897688
# Epoch   11/20 hypothesis: tensor([152.5485, 183.3610, 180.6640, 196.7389, 139.8602]) Cost: 1.710541
# Epoch   12/20 hypothesis: tensor([152.6620, 183.4982, 180.7988, 196.8857, 139.9651]) Cost: 1.651413
# Epoch   13/20 hypothesis: tensor([152.7253, 183.5752, 180.8742, 196.9678, 140.0240]) Cost: 1.632375
# Epoch   14/20 hypothesis: tensor([152.7606, 183.6184, 180.9164, 197.0138, 140.0571]) Cost: 1.625923
# Epoch   15/20 hypothesis: tensor([152.7802, 183.6427, 180.9399, 197.0395, 140.0759]) Cost: 1.623412
# Epoch   16/20 hypothesis: tensor([152.7909, 183.6565, 180.9530, 197.0538, 140.0865]) Cost: 1.622141
# Epoch   17/20 hypothesis: tensor([152.7968, 183.6643, 180.9603, 197.0618, 140.0927]) Cost: 1.621253
# Epoch   18/20 hypothesis: tensor([152.7999, 183.6688, 180.9644, 197.0662, 140.0963]) Cost: 1.620500
# Epoch   19/20 hypothesis: tensor([152.8014, 183.6715, 180.9666, 197.0686, 140.0985]) Cost: 1.619770
# Epoch   20/20 hypothesis: tensor([152.8020, 183.6731, 180.9677, 197.0699, 140.0999]) Cost: 1.619063

    
    ## 3. High-level Implementation with nn.Module
#0. LinearRegressionModel(nn.Module)
# # Basically, all models of PyTorch are created by inheriting the provided 'nn.Module'
# class LinearRegressionModel(nn.Module):
#     # __init__: define the layer to use
#     # nn.linear: linear regression model
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(1, 1)
# 
#     # forward: tells how to calculate the output value from the input value
#     def forward(self, x):
#         return self.linear(x)
# 
# model = LinearRegressionModel()

#  --> input dimension from 1 to 3    
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    
    def forward(self, x):
        return self.linear(x)
        
#data
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])    
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]]) 

#model initialization
# W = torch.zeros((3, 1), requires_grad = True)   
# b = torch.zeros(1, requires_grad = True)   
model = MultivariateLinearRegressionModel()

#setting optimizer
# optimizer = optim.SGD([W, b], lr=1e-5)
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

    #compute H(x)
    # hypothesis = x_train.matmul(W) + b      # or .mm or @
    prediction = model(x_train)

    #compute cost
    # cost = torch.mean((hypothesis - y_train) ** 2)
    cost = F.mse_loss(prediction, y_train)

    #improve H(x) using cost
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    #print log 
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()
        ))
# Epoch    0/20 Cost: 31667.597656
# Epoch    1/20 Cost: 9926.267578
# Epoch    2/20 Cost: 3111.514160
# Epoch    3/20 Cost: 975.451477
# Epoch    4/20 Cost: 305.908691
# Epoch    5/20 Cost: 96.042679
# Epoch    6/20 Cost: 30.260782
# Epoch    7/20 Cost: 9.641695
# Epoch    8/20 Cost: 3.178671
# Epoch    9/20 Cost: 1.152871
# Epoch   10/20 Cost: 0.517863
# Epoch   11/20 Cost: 0.318801
# Epoch   12/20 Cost: 0.256387
# Epoch   13/20 Cost: 0.236816
# Epoch   14/20 Cost: 0.230663
# Epoch   15/20 Cost: 0.228717
# Epoch   16/20 Cost: 0.228095
# Epoch   17/20 Cost: 0.227883
# Epoch   18/20 Cost: 0.227798
# Epoch   19/20 Cost: 0.227758
# Epoch   20/20 Cost: 0.227729

