import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import logging
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

"""
def get_logger(log_dir):

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(os.path.join(log_dir, "train")):
        os.mkdir(os.path.join(log_dir, "train"))

    current_Time = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
    fileHandler = logging.FileHandler(os.path.join(log_dir, "train", "train_xception_%s.log"%current_Time))
    fileHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)

    return logger
# end
"""

def train(gpu_available, epochs=200,
          num_batches_per_epoch=1000,
          batch_size=512,
          learning_rate=0.0008,
          n=16,
          show_every_n_steps=500):

    print("train")

    # define networks
    alice = CryptoNN(D_in=(n*2), H=(n*2))
    bob = CryptoNN(D_in=(n*2), H=(n*2))
    eve = CryptoNN(D_in=n, H=(n*2))

    if gpu_available:
        alice.cuda()
        bob.cuda()
        eve.cuda()

    # aggregate errors
    bob_training_errors = []
    eve_training_errors = []

    # define optimizers
    optimizer_alice_bob = Adam(params=list(alice.parameters()) + list(bob.parameters()), lr=learning_rate)
    optimizer_eve = Adam(params=eve.parameters(), lr=learning_rate)

    # define losses 
    reconstruction_error_bob = nn.L1Loss()
    reconstruction_error_eve = nn.L1Loss()

    # training loop
    for epoch in range(epochs):
        bob_decrypt_error = 1.0
        eve_decrypt_error = 1.0

        for network in ["alice_bob", "eve"]:

            #if network == "eve":
            #    num_batches_per_epoch *= 2

            for step in range(num_batches_per_epoch):

                p, k = generate_data(gpu_available=gpu_available, batch_size=batch_size, n=n)

                alice_c = alice.forward(torch.cat((p, k), 1).float())
                bob_p = bob.forward(torch.cat((alice_c, k), 1).float())

                if network == "alice_bob":
                    error_bob = reconstruction_error_bob(input=bob_p, target=p)
                    optimizer_alice_bob.zero_grad()
                    error_bob.backward()
                    optimizer_alice_bob.step()
                
                if step % show_every_n_steps == 0:
                    print("Epoch:{} || Training:{} || Step:{} of {} || Loss:{}".format(epoch,
                                                                                       network,
                                                                                       step,
                                                                                       num_batches_per_epoch,
                                                                                       error_bob))
                if step == num_batches_per_epoch-1:
                    print("p[0]: {} \n".format(p[0]))
                    print("alice_c[0]: {} \n".format(alice_c[0]))
                    print("bob_p[0]: {} \n".format(bob_p[0]))
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
        train(gpu_available=gpu_available)
# end

if __name__ == "__main__":
    main()
