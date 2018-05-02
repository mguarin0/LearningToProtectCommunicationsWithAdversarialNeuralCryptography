import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import argparse
import os
import numpy as np
import time
import itertools

from models import MixTransformNN
from utils import generate_data, prjPaths, UTF_8_to_binary, binary_to_UTF_8, persist_object, restore_persist_object

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Implementation of cryptogan")
    parser.add_argument("--run_type",
                         type=str,
                         default="train",
                         choices=["train", "inference"],
                         help="train model or load trained model for interence")
    parser.add_argument("--n",
                         type=int,
                         default=16,
                         help="length of plaintext (message length)")
    parser.add_argument("--training_steps",
                        type=int,
                        default=25000,
                        help="number of training steps")
    parser.add_argument("--batch_size",
                        type=int,
                        default=256,
                        help="number training examples per (mini)batch")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.0008,
                        help="learning rate")
    parser.add_argument("--show_every_n_steps",
                        type=int,
                        default=100,
                        help="during training print output to cli every n steps")
    parser.add_argument("--checkpoint_every_n_steps",
                        type=int,
                        default=5000,
                        help="checkpoint model files during training every n epochs")
    parser.add_argument("--verbose",
                        type=bool,
                        default=False,
                        help="during training print model outputs to cli")
    parser.add_argument("--clip_value",
                        type=float,
                        default=1,
                        help="maximum allowed value of the gradients in range(-clip_value, clip_value)")

    args = parser.parse_args()
    return args
# end


def train(gpu_available,
          prjPaths,
          n,
          training_steps,
          batch_size,
          learning_rate,
          show_every_n_steps,
          checkpoint_every_n_steps,
          verbose,
          clip_value,
          aggregated_losses_every_n_steps=100):

    # define networks
    alice = MixTransformNN(D_in=(n*2), H=(n*2))
    bob = MixTransformNN(D_in=(n*2), H=(n*2))
    eve = MixTransformNN(D_in=n, H=(n*2))

    # specify that model is currently in training mode
    alice.train()
    bob.train()
    eve.train()

    if gpu_available:
        alice.cuda()
        bob.cuda()
        eve.cuda()

    # pickle n (message length)
    persist_object(full_path=os.path.join(prjPaths.PERSIST_DIR, "n.p"), x=n)

    # aggregate training errors
    aggregated_losses = {
            "alice_bob_training_loss": [],
            "bob_reconstruction_training_errors": [],
            "eve_reconstruction_training_errors": [],
            "step": []
    }

    # define optimizers
    optimizer_alice = Adam(params=alice.parameters(), lr=learning_rate)
    optimizer_bob = Adam(params=bob.parameters(), lr=learning_rate)
    optimizer_eve = Adam(params=eve.parameters(), lr=learning_rate)

    # define losses 
    bob_reconstruction_error = nn.L1Loss()
    eve_reconstruction_error = nn.L1Loss()

    # training loop
    for step in range(training_steps+1):

        # start time for step
        tic = time.time()

        # Training alternates between Alice/Bob and Eve
        for network, num_minibatches in {"alice_bob": 1, "eve": 2}.items():

            """ 
            Alice/Bob training for one minibatch, and then Eve training for two minibatches this ratio 
            in order to give a slight computational edge to the adversary Eve without training it so much
            that it becomes excessively specific to the exact current parameters of Alice and Bob
            """
            for minibatch in range(num_minibatches):

                p, k = generate_data(gpu_available=gpu_available, batch_size=batch_size, n=n)

                # forward pass through alice and eve networks
                alice_c = alice.forward(torch.cat((p, k), 1).float())
                eve_p = eve.forward(alice_c)

                if network == "alice_bob":

                    # forward pass through bob network
                    bob_p = bob.forward(torch.cat((alice_c, k), 1).float())

                    # calculate errors
                    error_bob = bob_reconstruction_error(input=bob_p, target=p)
                    error_eve = eve_reconstruction_error(input=eve_p, target=p)
                    alice_bob_loss =  error_bob + (1.0 - error_eve**2)

                    # Zero gradients, perform a backward pass, clip gradients, and update the weights.
                    optimizer_alice.zero_grad()
                    optimizer_bob.zero_grad()
                    alice_bob_loss.backward()
                    nn.utils.clip_grad_value_(alice.parameters(), clip_value)
                    nn.utils.clip_grad_value_(bob.parameters(), clip_value)
                    optimizer_alice.step()
                    optimizer_bob.step()

                elif network == "eve":

                    # calculate error
                    error_eve = eve_reconstruction_error(input=eve_p, target=p)

                    # Zero gradients, perform a backward pass, and update the weights
                    optimizer_eve.zero_grad()
                    error_eve.backward()
                    nn.utils.clip_grad_value_(eve.parameters(), clip_value)
                    optimizer_eve.step()

        # end time time for step
        time_elapsed = time.time() - tic

        if step % aggregated_losses_every_n_steps == 0:
            # aggregate min training errors for bob and eve networks
            aggregated_losses["alice_bob_training_loss"].append(alice_bob_loss.cpu().detach().numpy().tolist())
            aggregated_losses["bob_reconstruction_training_errors"].append(error_bob.cpu().detach().numpy().tolist())
            aggregated_losses["eve_reconstruction_training_errors"].append(error_eve.cpu().detach().numpy().tolist())
            aggregated_losses["step"].append(step)

        if step % show_every_n_steps == 0:
            print("Total_Steps: %i of %i || Time_Elapsed_Per_Step: (%.3f sec/step) || Bob_Alice_Loss: %.5f || Bob_Reconstruction_Error: %.5f || Eve_Reconstruction_Error: %.5f" % (step,
                                                                                                                                                                                   training_steps,
                                                                                                                                                                                   time_elapsed,
                                                                                                                                                                                   aggregated_losses["alice_bob_training_loss"][-1],
                                                                                                                                                                                   aggregated_losses["bob_reconstruction_training_errors"][-1],
                                                                                                                                                                                   aggregated_losses["eve_reconstruction_training_errors"][-1]))

        if step % checkpoint_every_n_steps == 0 and step != 0:
            print("checkpointing models...\n")
            torch.save(alice.state_dict(), os.path.join(prjPaths.CHECKPOINT_DIR, "alice.pth"))
            torch.save(bob.state_dict(), os.path.join(prjPaths.CHECKPOINT_DIR, "bob.pth"))
            torch.save(eve.state_dict(), os.path.join(prjPaths.CHECKPOINT_DIR, "eve.pth"))

    # pickle aggregated list of errors
    persist_object(full_path=os.path.join(prjPaths.PERSIST_DIR, "aggregated_losses.p"), x=aggregated_losses)
# end

def inference(gpu_available, prjPaths):
    
    # declare function member constant
    NUM_BITS_PER_BYTE = 8

    # restore variable to describe message length used to determine network dimensions
    n = restore_persist_object(full_path=os.path.join(prjPaths.PERSIST_DIR, "n.p"))

    # define networks
    alice = MixTransformNN(D_in=(n*2), H=(n*2))
    bob = MixTransformNN(D_in=(n*2), H=(n*2))
    eve = MixTransformNN(D_in=n, H=(n*2))


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

    convert_tensor_to_list_and_scale = lambda tensor: list(map(lambda x: int((round(x)+1)/2), tensor.cpu().detach().numpy().tolist()))

    while True:

        p_utf_8 = input("enter plaintext: ")

        # ensure that p is correct length else pad with spaces
        while not ((len(p_utf_8) * NUM_BITS_PER_BYTE) % n == 0):
            p_utf_8 = p_utf_8 + " "

        # convert p UTF-8 -> Binary
        p_bs = UTF_8_to_binary(p_utf_8)

        print("plaintext ({}) in binary: {}".format(p_utf_8, p_bs))

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

            # run forward pass through networks
            alice_c = torch.unsqueeze(alice.forward(torch.cat((p_b, k), 1).float()), 0)
            eve_p = convert_tensor_to_list_and_scale(eve.forward(alice_c))
            bob_p = convert_tensor_to_list_and_scale(bob.forward(torch.cat((alice_c, k), 1).float()))

            eve_ps_b.append("".join(list(map(str, eve_p))))
            bob_ps_b.append("".join(list(map(str, bob_p))))
        
        print("eve_ps_b:                     {}".format(list(itertools.chain.from_iterable([[i[:8], i[8:]]  for i in eve_ps_b]))))
        print("bob_ps_b:                     {}\n".format(list(itertools.chain.from_iterable([[i[:8], i[8:]]  for i in bob_ps_b]))))

        # TODO if model isn't well trained then it will predict binary values that are not valid in UTF-8 find another way to demo
        #eve_p_utf_8 = binary_to_UTF_8(eve_ps_b)
        #bob_p_utf_8 = binary_to_UTF_8(bob_ps_b)

        #print("eve_p_utf_8: {}".format(eve_p_utf_8))
        #print("bob_p_utf_8: {}\n".format(bob_p_utf_8))
# end

def main():
    args = get_args()
    prjPaths_ = prjPaths()

    # determine if gpu present
    if torch.cuda.device_count() > 0:
        gpu_available = True
    else:
        gpu_available = False

    if args.run_type == "train":
        train(gpu_available=gpu_available,
              prjPaths=prjPaths_,
              n=args.n,
              training_steps=args.training_steps,
              batch_size=args.batch_size,
              learning_rate=args.learning_rate,
              show_every_n_steps=args.show_every_n_steps,
              checkpoint_every_n_steps=args.checkpoint_every_n_steps,
              verbose=args.verbose,
              clip_value=args.clip_value)
    elif args.run_type == "inference":
        inference(gpu_available, prjPaths=prjPaths_)
# end

if __name__ == "__main__":
    main()
