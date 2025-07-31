import torch
import os
import sys
x = torch.rand(5, 3)
# print(x)


thrifty_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'thrifty'))
print(thrifty_path)