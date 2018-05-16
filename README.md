# GAN Cryptosystem

## Overview
This repo contains a PyTorch implementation of the Google Brain paper [Learning to Protect Communications with Adversarial Neural Cryptography [1]](https://arxiv.org/pdf/1610.06918.pdf) which introduces a novel deep learning architecture to learn a symmetric encryption system.

The system is organized as three parties: Alice, Bob, and Eve in accordance with the classic symmetric cryptosystem probelm introduced by Rivest, et al. [[2]](https://people.csail.mit.edu/rivest/Rsapaper.pdf). Alice and Bob wish to communicate securely, and Eve wishes to eavesdrop on their communication. Desired security property is secrecy (not integrity) therefore Eve can intercept the communications but can do nothing more.

<figure>
<img src="assets/OverviewOfCryptosystem.png" height="800px" width="1000px" align="center">
<figcaption> Figure 1: Overview of Cryptosystem </figcaption>
</figure>

To model this problem we introduce the system displayed below in [Figure 1](assets/OverviewOfCryptosystem.png). Each of the three parties are themselves neural networks with different objectives. The Alice and Bob networks wish to communicate such that they are able to communicate with maximal clarity while also maximally hiding their communications from Eve, the eavsedropper in the system. To communicate with Bob, Alice sends a confidential message _P_ (plaintext) to Bob. The message _P_ is an input to Alice along with _K_ (key). When Alice processes the input _P_ and _K_ and produces _C_ the cipher text. Both the Bob and Eve networks receive _C_, in an attempt to recover _P_, the original message from Alice. The Bob and Eve networks output as _P<sub>Bob</sub>_ and _P<sub>Eve</sub>_, respectively. Alice and Bob share the same secrete key, _K_, which provides them an advantage over Eve. The secrete key, _K_, is regenerated for each plaintext.

The objectives of each of the network is as follows: Eve’s goal is to reconstruct _P_ accurately and therefore minimize the error between _P_ and _P<sub>Eve</sub>_.  Alice and Bob's goal(s) are to communicate clearly thus to minimize the error between _P_ and _P<sub>Bob</sub>_ but also to hide their communication from Eve. Eve’s objectives contrast with Alice and Bob’s therefore this problem is a great candidate for adversarial training. Alice and Bob discover a cryptosystem to achieve their objectives.

To review the networks objective in a more formal sense:
* Alice Network: _C = A (&theta;<sub>A</sub>, P, K)_
* Bob Network: _P<sub>Bob</sub> = B (&theta;<sub>B</sub>, C, K)_
* Eve Network: _P<sub>Eve</sub> = E (&theta;<sub>E</sub>, C)_
* L1 distance: _d (P, P <sup>prime</sup>) = &Sigma;<sub>i=1, N</sub> | P<sub>i</sub> - P<sub>i</sub><sup>prime</sup> |_ where N is the length of plaintexts.
* Bob Reconstruction Error: _L<sub>B</sub> (&theta;<sub>A</sub>, &theta;<sub>B</sub>, P, K) = d (P, B (&theta;<sub>B</sub>, A (&theta;<sub>A</sub>, P, K), K))_
* Eve Reconstruction Error: _L<sub>E</sub> (&theta;<sub>A</sub>, &theta;<sub>E</sub>, P, K) = d (P, E (&theta;<sub>E</sub>, A (&theta;<sub>A</sub>, P, K)))_
* Loss for Alice and Bob: _L<sub>AB</sub> (&theta;<sub>A</sub>, &theta;<sub>B</sub>) = L<sub>B</sub> (&theta;<sub>A</sub>, &theta;<sub>B</sub>, P, K) -  L<sub>E</sub> (&theta;<sub>A</sub>, &theta;<sub>E</sub>, P, K)_ the combination reflects that Alice and Bob want to minimize Bob’s reconstruction error and to maximize the reconstruction error.

## Files in this repo
* Main Interface to the program contains Model Training and Inference: [main.py](src/main.py)
* Definition of Mix and Transform Architecture: [models.py](src/models.py)
* Paths, Plaintext/Key Generation, Encoding/Decoding between UTF-8 and Binary: [utils.py](src/utils.py)
* Plots of training results: [plots.ipynb](src/plots.ipynb)

## Dependencies
* python 3
* PyTorch 0.4.0

## How to run
all commands below are assuming that current working directory is `LearningToProtectCommunicationsWithAdversarialNeuralCryptography/src`
`python3 main.py -h` get a list of all command line arguments

`python3 main.py` train model with default command line arguments

`python3 main.py --run_type inference` run model inference

## Network Details
The network architecture introduced by Adbadi, et al. [[1]](https://arxiv.org/pdf/1610.06918.pdf) is known as the Mix and Transform Architecture. All binary encoded plaintext bits are mapped to [-1,1]. Alice and Bob consists of 1 x Fully Connected Layer 2N x 2N where N is the length in bits of the message. The fully connected layer is then followed by 4 x 1D Convolutional Layers with filter sizes [4, 2, 1, 1], input channels [1, 2, 1, 1], output channels[2, 4, 4, 1]. The strides for the 1D convolution by layer are [1, 2, 1, 1]. Note that same convolution is used to all convolutional layers in order to keep input and output diminsions the same. The activation functions used at each layer are the Sigmoid for all layers except final layer which is a Tanh used to bring values back to a range [-1, 1] that can map to binary values. The Eve uses more or less the same architecture except the Fully Connected Layer dimensions are N x 2N where N is the length in bits of the message because only receiving _C_. It should be noted that _P_, _K_ are vectors of same size; however, there is no reason that _K_ has to be the same size as _P_. _P_ and _K_ are generated from uniform distribution and values are mapped from [0,1] to [-1,1] All Network parameters randomly initialized.

The optimization strategy is mini-batch Gradient Descent with the Adam Optimizer. We want to approximate optimal Eve and therefore alternate training between the Alice-Bob and Eve training, training Eve on 2 mini-batches per step in order to give advantage to the adversary.

## Results
<figure>
<img src="assets/EvolutionOfBobAndEveReconstructionErrors.png" height="800px" width="900px" align="center">
<figcaption> Figure 2: Evolution Of Bob And Eve Reconstruction Errors</figcaption>
</figure>

Bob and Eve accomplish their training goals within approximately 15,000 training steps. It can be observed that Bob's reconstruction error continues to improve while Eve's reconstruction error remains just slightly better than 1 or 2 bits better than random guessing. This shows that Bob's error is being minimized while Eve's error is being maximized.

<figure>
<img src="assets/AliceBobTotalError.png" height="800px" width="900px" align="center">
<figcaption> Figure 3: Alice Bob Total Error </figcaption>
</figure>

In the figure above it can be observed that Alice and Bob's total training error is improved throughout the training process. These findings were consistent with the results published in the original paper Adbadi, et al. [[1]](https://arxiv.org/pdf/1610.06918.pdf).

## Limitations of the Mix and Transform Architecture
The symmetric cryptosystem proposed by Adbadi, et al. [[1]](https://arxiv.org/pdf/1610.06918.pdf) shows very promising results for encrypting binary encoded plaintext messages; however, here are several implementation details that prevent this system from having practical application. Working on binary encoded plaintext messages is problematic because if the reconstructed plaintext has any bit that is not properly reconstructed it could result in a non valid binary string or could completely change the meaning of the original message. In the rehelm of deep learning working with bits does not provide the same computational advantage that it does with classic encryption algorithms. The neural networks proposed in this system could therefore operate over tokenized plaintext rather than binary encoded plaintext with the same computational complexity. In future work to improve the practical application of a deep learning based symmetric cryptosystem input to the system could be tokenized plaintext or word embedded vectors.

## References
1) Martin Abadi, David G. Andersen. [Learning To Protect Communications With Generative Adversarial Neural Networks](https://arxiv.org/pdf/1610.06918.pdf). arXiv:1610.06918, October 2016.
2) Rivest, R. L.; Shamir, A.; Adleman, L. [A Method for Obtaining Digital Signatures and Public-key Cryptosystems](https://people.csail.mit.edu/rivest/Rsapaper.pdf). January 1978.
