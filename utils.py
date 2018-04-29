import torch
from torch.autograd import Variable

# TODO make sure that these tensors run on the gpu
def generate_data(batch_size, n):
    return [Variable(torch.randint(0,100, (batch_size, n), dtype=torch.float),
                              requires_grad=True),
           Variable(torch.randint(0, 100, (batch_size, n), dtype=torch.float),
                             requires_grad=True)]