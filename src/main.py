import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import logging
import argparse
import os
import numpy as np

from models import CryptoNN
from utils import generate_data, prjPaths, UTF_8_to_binary, binary_to_UTF_8

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

def train(gpu_available,
          prjPaths,
          n=16,
          epochs=200,
          num_batches_per_epoch=1000,
          batch_size=512,
          learning_rate=0.0008,
          show_every_n_steps=500,
          checkpoint_every_n_epochs=5,
          verbose=False):

    # define networks
    alice = CryptoNN(D_in=(n*2), H=(n*2))
    bob = CryptoNN(D_in=(n*2), H=(n*2))
    eve = CryptoNN(D_in=n, H=(n*2))

    # specify that model is currently in training mode
    alice.train()
    bob.train()
    eve.train()

    if gpu_available:
        alice.cuda()
        bob.cuda()
        eve.cuda()

    # aggregate training errors
    bob_reconstruction_training_errors = []
    eve_reconstruction_training_errors = []

    # define optimizers
    optimizer_alice_bob = Adam(params=list(alice.parameters()) + list(bob.parameters()), lr=learning_rate)
    optimizer_eve = Adam(params=eve.parameters(), lr=learning_rate)

    # define losses 
    bob_reconstruction_error = nn.L1Loss()
    eve_reconstruction_error = nn.L1Loss()

    # training loop
    for epoch in range(epochs):
        bob_min_error_per_epoch = 1.0
        eve_min_error_per_epoch = 1.0

        for network, num_steps in {"alice_bob": num_batches_per_epoch,
                                   "eve": num_batches_per_epoch*2}.items():

            for step in range(num_steps):

                p, k = generate_data(gpu_available=gpu_available, batch_size=batch_size, n=n)

                alice_c = alice.forward(torch.cat((p, k), 1).float())
                eve_p = eve.forward(alice_c)

                if network == "alice_bob":
                    bob_p = bob.forward(torch.cat((alice_c, k), 1).float())

                if network == "alice_bob":
                    error_bob = bob_reconstruction_error(input=bob_p, target=p)
                    error_eve = eve_reconstruction_error(input=eve_p, target=p)
                    bob_loss =  error_bob + (((n/2) - error_eve**2)/(n/2)**2)
                    optimizer_alice_bob.zero_grad()
                    bob_loss.backward()
                    optimizer_alice_bob.step()
                    loss = bob_loss
                    bob_min_error_per_epoch = min(bob_min_error_per_epoch, error_bob)
                elif network == "eve":
                    error_eve = eve_reconstruction_error(input=eve_p, target=p)
                    optimizer_eve.zero_grad()
                    error_eve.backward()
                    optimizer_eve.step()
                    loss = error_eve
                    eve_min_error_per_epoch = min(eve_min_error_per_epoch, error_eve)

                if step % show_every_n_steps == 0:
                    print("Epoch_{}:{} || Training:{} || Step:{} of {} || {}_Loss:{}".format(network,
                                                                                             epoch,
                                                                                             network,
                                                                                             step,
                                                                                             num_steps,
                                                                                             network,
                                                                                             loss))

                if step == num_batches_per_epoch-1 and verbose:
                    print("p[0]: {} \n".format(p[0]))
                    print("alice_c[0]: {} \n".format(alice_c[0]))
                    if network == "alice_bob":
                        print("bob_p[0]: {} \n".format(bob_p[0]))
                    elif network == "eve":
                        print("eve_p[0]: {} \n".format(eve_p[0]))
                    print("\n")

        # aggregate min training errors for bob and eve networks
        bob_reconstruction_training_errors.append(bob_min_error_per_epoch)
        eve_reconstruction_training_errors.append(eve_min_error_per_epoch)

        if epoch % checkpoint_every_n_epochs == 0:
            torch.save(alice.state_dict(), os.path.join(prjPaths.CHECKPOINT_DIR, "alice.pth"))
            torch.save(bob.state_dict(), os.path.join(prjPaths.CHECKPOINT_DIR, "bob.pth"))
            torch.save(eve.state_dict(), os.path.join(prjPaths.CHECKPOINT_DIR, "eve.pth"))

# end


def inference(gpu_available, prjPaths, n):

    # declare function member constant
    NUM_BITS_PER_BYTE = 8

    # define networks
    alice = CryptoNN(D_in=(n*2), H=(n*2))
    bob = CryptoNN(D_in=(n*2), H=(n*2))
    eve = CryptoNN(D_in=n, H=(n*2))

    # restoring persisted networks
    print("restoring Alice, Bob, and Eve networks...\n")
    alice.load_state_dict(torch.load(os.path.join(prjPaths.CHECKPOINT_DIR, "alice.pth")))
    bob.load_state_dict(torch.load(os.path.join(prjPaths.CHECKPOINT_DIR, "bob.pth")))
    eve.load_state_dict(torch.load(os.path.join(prjPaths.CHECKPOINT_DIR, "eve.pth")))

    # specify that model is currently in training mode
    alice.eval()
    bob.eval()
    eve.eval()

    # if gpu available then run inference on gpu
    if gpu_available:
        alice.cuda()
        bob.cuda()
        eve.cuda()

    convert_tensor_to_list_and_scale = lambda tensor: list(map(lambda x: int((x+1)/2), tensor.cpu().detach().numpy().tolist()))


    while True:

        p_utf_8 = input("enter plaintext: ")

        # ensure that p is correct length else pad with spaces
        while not ((len(p_utf_8) * NUM_BITS_PER_BYTE) % n == 0):
            p_utf_8 = p_utf_8 + " "

        # convert p UTF-8 -> Binary
        p_bs = UTF_8_to_binary(p_utf_8)

        # group Binary p into groups that are valid with input layer of network
        p_bs = [np.asarray(list(p_bs[i-1]+p_bs[i]), dtype=np.float32) for i, p_b in enumerate(p_bs) if ((i-1) * NUM_BITS_PER_BYTE) % n == 0]

        eve_ps_b = []
        bob_ps_b = []
        for p_b in p_bs:

            # generate k
            _, k = generate_data(gpu_available=gpu_available, batch_size=1, n=n) 
            p_b = torch.unsqueeze(torch.from_numpy(p_b)*2-1, 0)

            if gpu_available:
                p_b = p_b.cuda()

            alice_c = torch.unsqueeze(alice.forward(torch.cat((p_b, k), 1).float()), 0)
            eve_p = convert_tensor_to_list_and_scale(eve.forward(alice_c))
            bob_p = convert_tensor_to_list_and_scale(bob.forward(torch.cat((alice_c, k), 1).float()))

            eve_ps_b.append("".join(list(map(str, eve_p))))
            bob_ps_b.append("".join(list(map(str, bob_p))))
            
        eve_p_utf_8 = binary_to_UTF_8(eve_ps_b)
        bob_p_utf_8 = binary_to_UTF_8(bob_ps_b)

        print("eve_p_utf_8: {}".format(eve_p_utf_8))
        print("bob_p_utf_8: {}\n".format(bob_p_utf_8))
# end

def main():
    args = get_args()
    prjPaths_ = prjPaths()

    if torch.cuda.device_count() > 0:
        gpu_available = True
    else:
        gpu_available = False

    if args.run_type == "train":
        train(gpu_available=gpu_available,
              prjPaths=prjPaths_)
    elif args.run_type == "inference":
        inference(gpu_available, prjPaths=prjPaths_, n=16)
# end

if __name__ == "__main__":
    main()
