import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import os

from models import CryptoNN
from utils import generate_data

"""
TODO: network weight initialization
"""
def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Implementation of cryptogan")
    parser.add_argument('--run_type',
                         type=str,
                         default="train",
                         choices=["train", "inference"],
                         help="train model or load trained model for interence")
    parser.add_argument('--msg_length',
                         type=int,
                         default=16,
                         help="length of plaintext (message length)")
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0008,
                        help="learning rate")
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help="number of training epochs")
    parser.add_argument('--minibatch_size',
                        type=int,
                        default=4096,
                        help="number training examples per minibatch")
    args = parser.parse_args()

    return args
# end

def train(epochs=10, num_batches_per_epoch=1000, batch_size=30, n=16):
    print("train")

    bob_training_errors = []
    eve_training_errors = []

    alice = CryptoNN(D_in=(n*2), H=n)
    bob = CryptoNN(D_in=(n*2), H=n)
    eve = CryptoNN(D_in=(n*2), H=n)

    for epoch in range(epochs):

        for network in ["alice_bob", "eve"]:

            if network == "eve":
                num_batches_per_epoch *= 2

            for i in range(num_batches_per_epoch):

                p, k = generate_data(batch_size=batch_size, n=n)
                x_prime_alice = alice.forward(torch.cat((p, k), 1).float())
                #print("x_prime_alice: {}".format(x_prime_alice))
                #print(x_prime_alice.shape)
                exit(0)

# end

def validate():
    print("validate")
# end

def test():
    print("test")
# end

def main():
    args = get_args()

    if torch.cuda.device_count() > 0:
        gpu_available = True
    else:
        gpu_available = False

    if args.run_type == "train":
        train()
# end

if __name__ == "__main__":
    main()
