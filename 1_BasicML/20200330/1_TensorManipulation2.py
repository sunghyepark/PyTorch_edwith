    # <2020.03.29 (Sun)>

import numpy as np
import torch

    ## (1) View (Reshape in numpy) *************************
t = np.array([[[0, 1, 2],
               [3, 4, 5]],

              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
print(ft.shape)
# size([2, 2, 3])  

print(ft.view([-1, 3]))         #reshape: (2, 2, 3) -> (?, 3) = (2x2, 3)
print(ft.view([-1, 3]).shape)
# tensor([[ 0.,  1.,  2.],
#         [ 3.,  4.,  5.],
#         [ 6.,  7.,  8.],
#         [ 9., 10., 11.]])
#torch.Size([4, 3])

print(ft.view([-1, 1, 3]))      #reshape: (2, 2, 3) -> (?, 1, 3) = (2x2, 1, 3)
print(ft.view([-1, 1, 3]).shape)
# tensor([[[ 0.,  1.,  2.]],
# 
#         [[ 3.,  4.,  5.]],
# 
#         [[ 6.,  7.,  8.]],
# 
#         [[ 9., 10., 11.]]])
# torch.Size([4, 1, 3])


    ## (2) Squeeze (automatically eliminates dimension with only 1 element)
print()
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)
# tensor([[0.],
#         [1.],
#         [2.]])
# torch.Size([3, 1])

print(ft.squeeze())         # (3, 1) -> (3)
print(ft.squeeze().shape)
# tensor([0., 1., 2.])
# torch.Size([3])

## e.g.) ft.squeeze(dim=0) -> (3,1)
## e.g.) ft.squeeze(dim=1) -> (3)


    ## (3) Unsqueeze (add 1 dimension)
print()
ft = torch.FloatTensor([0, 1, 2])
print(ft.shape)
# torch.Size([3])

print(ft.unsqueeze(0))          # (3) -> (1, 3)
print(ft.unsqueeze(0).shape)
# tensor([0., 1., 2.])
# torch.Size([3])

print(ft.view(1, -1))           # (3) -> (1, ?)
print(ft.view(1, -1).shape)
# tensor([[0., 1., 2.]])
# torch.Size([1, 3])

print(ft.unsqueeze(1))          # (3) -> (3, 1)
print(ft.unsqueeze(1).shape)
# tensor([[0.],
#         [1.],
#         [2.]])
# torch.Size([3, 1])

print(ft.unsqueeze(-1))         # (3) -> (3, 1)
print(ft.unsqueeze(-1).shape)
# tensor([[0.],
#         [1.],
#         [2.]])
# torch.Size([3, 1])


    ## (4) Scatter (for one-hot encoding)
print()
lt = torch.LongTensor([[0], [1], [2], [0]])
print(lt)
# tensor([[0],
#         [1],
#         [2],
#         [0]])

one_hot = torch.zeros(4, 3)      # batch_size = 4, classes = 3
one_hot.scatter_(1, lt, 1)
print(one_hot)
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.],
#         [1., 0., 0.]])


    ## (5) Type casting (long <-> float)
print()
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
# tensor([1, 2, 3, 4])

print(lt.float())               # long -> float
# tensor([1., 2., 3., 4.])

bt = torch.ByteTensor([True, False, False, True])       #(1, 0, 0, 1)
print(bt)
# tensor([1, 0, 0, 1], dtype=torch.uint8)

print(bt.long())
print(bt.float())
# tensor([1, 0, 0, 1])
# tensor([1., 0., 0., 1.])


    ## (6) Concatenation
print()
x = torch.FloatTensor([[1, 2], [3, 4]])         # (2, 2)
y = torch.FloatTensor([[5, 6], [7, 8]])         # (2, 2)

print(torch.cat([x, y], dim = 0))               # (4, 2)
# tensor([[1., 2.],
#         [3., 4.],
#         [5., 6.],
#         [7., 8.]])
print(torch.cat([x, y], dim = 1))               # (2, 4)
# tensor([[1., 2., 5., 6.],
#         [3., 4., 7., 8.]])


    ## (7) Stacking
print()
x = torch.FloatTensor([1, 4])       # (2)
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))       # (3, 2)
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])
print(torch.stack([x, y, z], dim = 1))  # (2, 3)
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim = 0))
        # (1,2) (1,2) (1,2) -> (3,2)
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])


    ## (8) Ones and Zeros Like
print()
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]]) 
print(x)                    # (2, 3)
# tensor([[0., 1., 2.],
#         [2., 1., 0.]])

print(torch.ones_like(x))   # (2, 3): all 1
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])
print(torch.zeros_like(x))  # (2, 3): all 0
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])


    ## (9) In-place Operation (_)
print()
x = torch.FloatTensor([[1, 2], [3, 4]])

print(x.mul(2.))
# tensor([[2., 4.],
#         [6., 8.]])
print(x)                    # x is not changed
# tensor([[1., 2.],
#         [3., 4.]])

print(x.mul_(2.))           # x is changed
# tensor([[2., 4.],
#         [6., 8.]])
print(x)
# tensor([[2., 4.],
#         [6., 8.]])


    ## (10) Zip
print()
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)
# 1 4
# 2 5
# 3 6
for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
# 1 4 7
# 2 5 8
# 3 6 9

