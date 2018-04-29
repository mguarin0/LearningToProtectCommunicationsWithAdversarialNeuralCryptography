import torch
from torch.autograd import Variable

# TODO make sure that these tensors run on the gpu
def generate_data(gpu_available, batch_size, n):
    if gpu_available:
        return [torch.randint(0, 2, (batch_size, n), dtype=torch.float).cuda()*2-1,
                torch.randint(0, 2, (batch_size, n), dtype=torch.float).cuda()*2-1]
    else:
        return [torch.randint(0, 2, (batch_size, n), dtype=torch.float)*2-1,
                torch.randint(0, 2, (batch_size, n), dtype=torch.float)*2-1]
# end