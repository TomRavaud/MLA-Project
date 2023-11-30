# 'Advanced Machine Learning' Project
*In French: Machine Learning Avancé (MLA)*

The goal of this project is to reproduce the results of a recent research paper in the field of Deep Learning (DL). Our group has chosen the paper [**"Net2Net: Accelerating Learning via Knowledge Transfer"**](https://arxiv.org/abs/1511.05641) by Chen et al. (2016) which introduces **Net2Net** techniques, aiming at transferring quickly the knowledge from a previously trained network to a wider or deeper one.

## Quick Start

To create and activate a virtual **conda** environment, run the following commands:

```bash
conda create -n mla python=3.10
conda activate mla
```

Then, install a version of **PyTorch** with CUDA support (if you have an NVIDIA GPU), compatible with your version of NVIDIA GPU driver. Have a look at the [PyTorch website](https://pytorch.org/) to find the right command to run. For instance, if you have an NVIDIA GPU with CUDA 11.8 support, run the following command:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Install the other dependencies by running the following command in the root directory of the project:

```bash
pip3 install -r requirements.txt
```

To execute the Jupyter notebooks, you will need to install the **ipykernel** package in the conda environment and to create
a kernel for this environment:
    
```bash
pip3 install ipykernel
python -m ipykernel install --user --name=mla
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


## TODO

- [x] Update the "Quick Start" section of the README.md file to help the user install PyTorch with CUDA support for a different version of NVIDIA GPU driver.
- [ ] Gather the two Net2Net techniques in a single class, and create a package for it.
- [x] Apply Net2WiderNet to Inception-V2 (need to adapt the code to take batch normalization and concatenation into account).
- [ ] Apply Net2DeeperNet to Inception-V2.
- [ ] Establish the pipeline to reproduce the results of the paper and set up the experiments on the university's GPU cluster.
- [ ] Update the README files in the `src` directory to describe the code.
- [x] Download the ImageNet dataset and train Inception-V2 from scratch on it.
- [x] Add dropout to the Inception-V2 architecture (or some random noise to the replicated weights) to help the student network to learn to use its full capacity.
- [x] Implement the "Random pad" baseline method to compare the Net2WiderNet technique with it.
- [x] Introduce a multiplicative factor to modulate the number of output filters of each branch of the Inception module, and check that setting this factor to $\sqrt{3}$ (as in the paper) leads to a reduction of 60% of the number of parameters.
- [x] Split the code of the Net2WiderNet technique into several functions to make it more readable.