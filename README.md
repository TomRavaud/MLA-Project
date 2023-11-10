# 'Advanced Machine Learning' Project
*In French: Machine Learning Avancé (MLA)*

The goal of this project is to reproduce the results of a recent research paper in the field of Deep Learning (DL). Our group has chosen the paper [**"Net2Net: Accelerating Learning via Knowledge Transfer"**](https://arxiv.org/abs/1511.05641) by Chen et al. (2016) which introduces **Net2Net** techniques, aiming at transferring quickly the knowledge from a previously trained network to a wider or deeper one.

## Quick Start

To create a virtual **conda** environment and install the required packages, run the following commands in the root directory of the project:

```bash
conda create -n mla python=3.10
conda activate mla
pip install -r requirements.txt
```

To exit the virtual environment, and remove it if needed, run the following commands:

```bash
conda deactivate
conda remove -n mla --all
```

## Net2Net

### Context of the study

When it comes to developing a neural network to perform a specific task, it is common to start with a relatively simple network architecture and then make it more complex to achieve better performance. In such a workflow, each new architecture is generally trained from scratch, and does not take advantage of what has been learned from previous architectures. This is costly in both time and money. To address this problem, the authors of this paper propose a strategy for transferring information learned by a neural network to a larger neural network. In this way, the latter takes less time to train.

Beyond this application, the authors also raise the idea of using their approach to develop lifelong learning systems. Indeed, it is common to seek to increase the capabilities of an existing neural network by training it on a larger database, and it is essential in this case to increase the complexity of the model to capture the larger distribution of data. Again, rather than training a new, more complex architecture from scratch, it is better to take advantage of the knowledge acquired by the previous model.


### Main idea of the paper

The main idea of the paper is "function-preserving initializations". In other words, after training a teacher network, we aim to create a more complex student network, whose weight initialization is such that it produces the same outputs as the teacher network. So, by updating the weights as we move in the direction of the gradient, we are guaranteed to have a student network at least as good as the teacher network.


### Proposed techniques

The authors propose two techniques to increase the complexity of a neural network. The first, called *Net2WiderNet*, makes it possible to increase the number of neurons in a layer (or equivalently with CNNs, the number of filters per convolution), without modifying the network predictions (this method can then be applied at several layers of the network). The second, called *Net2DeeperNet*, allows you to increase the number of layers of the network, without modifying the network predictions. These two techniques can be combined to obtain a student network that is wider and deeper than the teacher network.