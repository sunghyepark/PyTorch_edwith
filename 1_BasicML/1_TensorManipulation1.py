    # <2020.03.28 (Sat)>

import numpy as np
import torch

    ## 1D Array with NumPy
#t = np.array([0., 1., 2., 3., 4., 5., 6.])
#print(t)    # [0. 1. 2. 3. 4. 5. 6.]
#
#print('Rank of t: ', t.ndim)    # 1
#print('Shape of t: ', t.shape)  # (7,)
#
#print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1])   # Element
#print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1])      # Slicing
#print('t[:2] t[3:] = ', t[:2], t[3:])            # Slicing
    ## 2D Array with NumPy
#t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
#print(t)   
# # [[1. 2. 3.]
# #  [4. 5. 6.]
# #  [7. 8. 9.]
# #  [10. 11. 12.]]
#
#print('Rank of t: ', t.ndim)    # 2
#print('Shape of t: ', t.shape)  # (4, 3)
    
    
    ## (1) 1D Array with pyTorch
print()
t1 = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t1)    # tensor([0., 1., 2., 3., 4., 5., 6.])

print(t1.dim())    # rank    -> 1
print(t1.shape)    # shape   -> torch.Size([7])
print(t1.size())   # shape   -> torch.Size([7])

print('t[0] t[1] t[-1] = ', t1[0], t1[1], t1[-1])   # Element      -> tensor(0.) tensor(1.) tensor(6.)
print('t[2:5] t[4:-1] = ', t1[2:5], t1[4:-1])      # Slicing
print('t[:2] t[3:] = ', t1[:2], t1[3:])            # Slicing
    
    
    ## (2) 2D Array with pyTorch
print()
t2 = torch.FloatTensor([[1., 2., 3.],
                        [4., 5., 6.],
                        [7., 8., 9.],
                        [10., 11., 12.]
                       ])
print(t2)    
# tensor([[1., 2., 3.],
#         [4., 5., 6.],
#         [7., 8., 9.],
#         [10., 11., 12.]])

print(t2.dim())    # rank    -> 2
print(t2.size())   # shape   -> torch.Size([4, 3])

print(t2[:, 1])          #tensor([ 2., 5., 8., 11.])   
print(t2[:, 1].size())   #torch.Size([4])
print(t2[:, :-1])
# tensor([[1., 2.],
#         [4., 5.],
#         [7., 8.],
#         [10., 11.]])


    ## (3) Braoadcasting: automatically resize
# Same shape
print()
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)   # tensor([[5., 5.]])

# Vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3]])       # 3 -> [[3, 3]]
print(m1 + m2)   # tensor([[4., 5.]])

# 2x1 Vector + 1x2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])      
print(m1 + m2)  
# tensor([[4., 5.],
#         [5., 6.]])


    ## (4) Multiplication vs Matrix Manipulation
print()
print('-------------')
print('Mul vs Matmul')
print('-------------')
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1.matmul(m2)) # 2 x 1
# tensor([[ 5.],
#         [ 11.]])

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1 * m2) # 2 x 2 (broadcasting)
# --> element wise mul
# tensor([[ 1., 2.],
#         [ 6., 8.]])
print(m1.mul(m2)) # 2 x 1
# tensor([[ 1., 2.],
#         [ 6., 8.]])


    ## (5) Mean
print()
t = torch.FloatTensor([1, 2])
print(t.mean())     # tensor(1.5000)

# Can't use mean() on integers
t = torch.LongTensor([1, 2])
try:
    print(t.mean())
except Exception as exc:
    print(exc)

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
# tensor([[ 1., 2.],
#         [ 3., 4.]])
print(t.mean())          # tensor(2.5000)
print(t.mean(dim=0))     # tensor([2., 3.])
print(t.mean(dim=1))     # tensor([1.5000, 3.5000])
print(t.mean(dim=-1))    # tensor([1.5000, 3.5000])


    ## (6) Sum
print()
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
# tensor([[ 1., 2.],
#         [ 3., 4.]])
print(t.sum())          # tensor(10.)
print(t.sum(dim=0))     # tensor([4., 6.])
print(t.sum(dim=1))     # tensor([3., 7.])
print(t.sum(dim=-1))    # tensor([3., 7.])


    ## (7) Max and Argmax
print()
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)
# tensor([[ 1., 2.],
#         [ 3., 4.]])

print(t.max)            # tensor(4.)

print(t.max(dim=0))     # Returns two values: max and argmax(index)
# (tensor([3., 4.]), tensor([1, 1]))
print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])

print(t.max(dim=1))
print(t.max(dim=-1))


