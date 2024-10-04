import torch

patch=(3,3)
x=torch.arange(16).float()
print(x, x.shape)
x2d = x.reshape(1,1,4,4)
print(x2d, x2d.shape)
h,w = patch
c=x2d.size(1)
print(c) # channels
# unfold(dimension, size, step)
r = x2d.unfold(2,h,1).unfold(3,w,1)
print(r.shape)
print(r) # result