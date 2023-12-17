"""
This file contains the parameters used to train the Inception-V2 network on
the CIFAR-10 dataset.
"""

# Import libraries
import numpy as np


# Dataset name (for logging purposes, can be anything)
DATASET_NAME = "CIFAR-10"

# Path to the dedicated directory
PATH = "src/inceptionv2_cifar/"

# Study name
STUDY_NAME = "test"


###############################################
## Architecture parameters (teacher network) ##
###############################################

# Inception modulation factor (how much the number of filters is multiplied)
INCEPTION_FACTOR = np.sqrt(0.3)

# Set the number of classes in the dataset
NB_CLASSES = 10


################################
## Parameters to train the NN ##
################################

# Learning parameters
LEARNING = {"batch_size": 128,
            "nb_epochs": 1,
            "learning_rate": 0.005,
            "weight_decay": 0.0001,
            "momentum": 0.9  # Only used for SGD
            }

# Learning rate that has been optimized for the teacher network
LR_TEACHER = 0.0015  # inception factor = sqrt(0.3)

# Learning rate that has been optimized on the random initialization baseline
LR_RANDOM_INITIALIZATION = 0.002

# Learning rate that has been optimized on Net2WiderNet
LR_NET2WIDERNET = 5e-5

# Learning rate that has been optimized on the random pad baseline
LR_RANDOM_PAD = 2e-4

# Number of random seeds to use to train the network
NB_SEEDS = 3

# Whether to train the student network or the teacher network
TRAIN_STUDENT = True


###################################
## Images' transforms parameters ##
###################################

# Shape of the images in the ImageNet dataset
IMAGE_SHAPE = (33, 33)
NB_CHANNELS = 3

# Define the mean and std of the dataset
# (pre-computed on the CIFAR-10 dataset)
NORMALIZE_PARAMS = {"mean": (0.4914, 0.4822, 0.4465),
                    "std": (0.247, 0.243, 0.261)}


############
## Optuna ##
############

# Study database storage path
STUDY_DB_PATH =\
    f"sqlite:///src/inceptionv2_cifar/logs/{STUDY_NAME}/{STUDY_NAME}.db"

# Optimize the student network (otherwise optimize the teacher network)
OPTIMIZE_STUDENT = True

# Number of trials for the Optuna optimization
NB_TRIALS = 3

# Number of seeds to test for each trial (in order to get more robust results)
NB_SEEDS_OPTUNA = 3

# Learning rate search space
LR_RANGE = (1e-5, 1e-2)

# Number of epochs to wait before stopping the training if the validation
# accuracy does not improve
PATIENCE = 5

###################
## Net2DeeperNet ##
###################
#List of layers to deepen
deeper_operations = {"operation1": {"target_conv_layers": ["net.2.0.b1.0","net.2.0.b2.0","net.2.0.b2.3","net.2.0.b3.0","net.2.0.b3.3","net.2.0.b4.1",
                                                           "net.2.1.b1.0","net.2.1.b2.0","net.2.1.b2.3","net.2.1.b3.0","net.2.1.b3.3","net.2.1.b4.1",
                                                           "net.3.0.b1.0","net.3.0.b2.0","net.3.0.b2.3","net.3.0.b3.0","net.3.0.b3.3","net.3.0.b4.1",
                                                           "net.3.1.b1.0","net.3.1.b2.0","net.3.1.b2.3","net.3.1.b3.0","net.3.1.b3.3","net.3.1.b4.1",
                                                           "net.3.2.b1.0","net.3.2.b2.0","net.3.2.b2.3","net.3.2.b3.0","net.3.2.b3.3","net.3.2.b4.1",
                                                           "net.3.3.b1.0","net.3.3.b2.0","net.3.3.b2.3","net.3.3.b3.0","net.3.3.b3.3","net.3.3.b4.1",
                                                           "net.3.4.b1.0","net.3.4.b2.0","net.3.4.b2.3","net.3.4.b3.0","net.3.4.b3.3","net.3.4.b4.1",
                                                           "net.4.0.b1.0","net.4.0.b2.0","net.4.0.b2.3","net.4.0.b3.0","net.4.0.b3.3","net.4.0.b4.1",
                                                           "net.4.1.b1.0","net.4.1.b2.0","net.4.1.b2.3","net.4.1.b3.0","net.4.1.b3.3","net.4.1.b4.1"]}}
# list of batchnorm to add for each layer deepen
deeper_batchnorm ={"net.2.0.b1.0": "net.2.0.b1.4","net.2.0.b2.0":"net.2.0.b2.4","net.2.0.b2.3":"net.2.0.b2.10","net.2.0.b3.0":"net.2.0.b3.4","net.2.0.b3.3":"net.2.0.b3.10","net.2.0.b4.1":"net.2.0.b4.5",
                    "net.2.1.b1.0": "net.2.1.b1.4","net.2.1.b2.0":"net.2.1.b2.4","net.2.1.b2.3":"net.2.1.b2.10","net.2.1.b3.0":"net.2.1.b3.4","net.2.1.b3.3":"net.2.1.b3.10","net.2.1.b4.1":"net.2.1.b4.5",
                    "net.3.0.b1.0": "net.3.0.b1.4","net.3.0.b2.0":"net.3.0.b2.4","net.3.0.b2.3":"net.3.0.b2.10","net.3.0.b3.0":"net.3.0.b3.4","net.3.0.b3.3":"net.3.0.b3.10","net.3.0.b4.1":"net.3.0.b4.5",
                    "net.3.1.b1.0": "net.3.1.b1.4","net.3.1.b2.0":"net.3.1.b2.4","net.3.1.b2.3":"net.3.1.b2.10","net.3.1.b3.0":"net.3.1.b3.4","net.3.1.b3.3":"net.3.1.b3.10","net.3.1.b4.1":"net.3.1.b4.5",
                    "net.3.2.b1.0": "net.3.2.b1.4","net.3.2.b2.0":"net.3.2.b2.4","net.3.2.b2.3":"net.3.2.b2.10","net.3.2.b3.0":"net.3.2.b3.4","net.3.2.b3.3":"net.3.2.b3.10","net.3.2.b4.1":"net.3.2.b4.5",
                    "net.3.3.b1.0": "net.3.3.b1.4","net.3.3.b2.0":"net.3.3.b2.4","net.3.3.b2.3":"net.3.3.b2.10","net.3.3.b3.0":"net.3.3.b3.4","net.3.3.b3.3":"net.3.3.b3.10","net.3.3.b4.1":"net.3.3.b4.5",
                    "net.3.4.b1.0": "net.3.4.b1.4","net.3.4.b2.0":"net.3.4.b2.4","net.3.4.b2.3":"net.3.4.b2.10","net.3.4.b3.0":"net.3.4.b3.4","net.3.4.b3.3":"net.3.4.b3.10","net.3.4.b4.1":"net.3.4.b4.5",
                    "net.4.0.b1.0": "net.4.0.b1.4","net.4.0.b2.0":"net.4.0.b2.4","net.4.0.b2.3":"net.4.0.b2.10","net.4.0.b3.0":"net.4.0.b3.4","net.4.0.b3.3":"net.4.0.b3.10","net.4.0.b4.1":"net.4.0.b4.5",
                    "net.4.1.b1.0": "net.4.1.b1.4","net.4.1.b2.0":"net.4.1.b2.4","net.4.1.b2.3":"net.4.1.b2.10","net.4.1.b3.0":"net.4.1.b3.4","net.4.1.b3.3":"net.4.1.b3.10","net.4.1.b4.1":"net.4.1.b4.5"}

##################
## Net2WiderNet ##
##################

# Random noise to add to the weights of the new layers
SIGMA = 0.001

# Use the baseline method (random pad) instead of Net2WiderNet
RANDOM_PAD = True

# Define the parameters of the Net2WiderNet algorithm for each call
# (there are three calls per Inception block)
# (these values are used to recover the structure of the original Inception-V2
# network from a version of the network with narrower Inception blocks)
wider_operations ={
    # Inception block 1
    "inception1.1": {"target_conv_layers": ["net.2.0.b2.0"],
                     "width": [96],
                     "batch_norm_layers": ["net.2.0.b2.1"],
                     "next_layers": ["net.2.0.b2.3"]},
    "inception1.2": {"target_conv_layers": ["net.2.0.b3.0"],
                     "width": [16],
                     "batch_norm_layers": ["net.2.0.b3.1"],
                     "next_layers": ["net.2.0.b3.3"]},
    "inception1.3": {"target_conv_layers": ["net.2.0.b1.0",
                                            "net.2.0.b2.3",
                                            "net.2.0.b3.3",
                                            "net.2.0.b4.1"],
                     "width": [64, 128, 32, 32],
                     "batch_norm_layers": ["net.2.0.b1.1",
                                           "net.2.0.b2.4",
                                           "net.2.0.b3.4",
                                           "net.2.0.b4.2"],
                     "next_layers": ["net.2.1.b1.0",
                                     "net.2.1.b2.0",
                                     "net.2.1.b3.0",
                                     "net.2.1.b4.1"]},
    # Inception block 2
    "inception2.1": {"target_conv_layers": ["net.2.1.b2.0"],
                     "width": [128],
                     "batch_norm_layers": ["net.2.1.b2.1"],
                     "next_layers": ["net.2.1.b2.3"]},
    "inception2.2": {"target_conv_layers": ["net.2.1.b3.0"],
                     "width": [32],
                     "batch_norm_layers": ["net.2.1.b3.1"],
                     "next_layers": ["net.2.1.b3.3"]},
    "inception2.3": {"target_conv_layers": ["net.2.1.b1.0",
                                            "net.2.1.b2.3",
                                            "net.2.1.b3.3",
                                            "net.2.1.b4.1"],
                     "width": [128, 192, 96, 64],
                     "batch_norm_layers": ["net.2.1.b1.1",
                                           "net.2.1.b2.4",
                                           "net.2.1.b3.4",
                                           "net.2.1.b4.2"],
                     "next_layers": ["net.3.0.b1.0",
                                     "net.3.0.b2.0",
                                     "net.3.0.b3.0",
                                     "net.3.0.b4.1"]},
    # Inception block 3
    "inception3.1": {"target_conv_layers": ["net.3.0.b2.0"],
                     "width": [96],
                     "batch_norm_layers": ["net.3.0.b2.1"],
                     "next_layers": ["net.3.0.b2.3"]},
    "inception3.2": {"target_conv_layers": ["net.3.0.b3.0"],
                     "width": [16],
                     "batch_norm_layers": ["net.3.0.b3.1"],
                     "next_layers": ["net.3.0.b3.3"]},
    "inception3.3": {"target_conv_layers": ["net.3.0.b1.0",
                                            "net.3.0.b2.3",
                                            "net.3.0.b3.3",
                                            "net.3.0.b4.1"],
                     "width": [192, 208, 48, 64],
                     "batch_norm_layers": ["net.3.0.b1.1",
                                           "net.3.0.b2.4",
                                           "net.3.0.b3.4",
                                           "net.3.0.b4.2"],
                     "next_layers": ["net.3.1.b1.0",
                                     "net.3.1.b2.0",
                                     "net.3.1.b3.0",
                                     "net.3.1.b4.1"]},
    # Inception block 4
    "inception4.1": {"target_conv_layers": ["net.3.1.b2.0"],
                     "width": [112],
                     "batch_norm_layers": ["net.3.1.b2.1"],
                     "next_layers": ["net.3.1.b2.3"]},
    "inception4.2": {"target_conv_layers": ["net.3.1.b3.0"],
                     "width": [24],
                     "batch_norm_layers": ["net.3.1.b3.1"],
                     "next_layers": ["net.3.1.b3.3"]},
    "inception4.3": {"target_conv_layers": ["net.3.1.b1.0",
                                            "net.3.1.b2.3",
                                            "net.3.1.b3.3",
                                            "net.3.1.b4.1"],
                     "width": [160, 224, 64, 64],
                     "batch_norm_layers": ["net.3.1.b1.1",
                                           "net.3.1.b2.4",
                                           "net.3.1.b3.4",
                                           "net.3.1.b4.2"],
                     "next_layers": ["net.3.2.b1.0",
                                     "net.3.2.b2.0",
                                     "net.3.2.b3.0",
                                     "net.3.2.b4.1"]},
    # Inception block 5
    "inception5.1": {"target_conv_layers": ["net.3.2.b2.0"],
                     "width": [128],
                     "batch_norm_layers": ["net.3.2.b2.1"],
                     "next_layers": ["net.3.2.b2.3"]},
    "inception5.2": {"target_conv_layers": ["net.3.2.b3.0"],
                     "width": [24],
                     "batch_norm_layers": ["net.3.2.b3.1"],
                     "next_layers": ["net.3.2.b3.3"]},
    "inception5.3": {"target_conv_layers": ["net.3.2.b1.0",
                                            "net.3.2.b2.3",
                                            "net.3.2.b3.3",
                                            "net.3.2.b4.1"],
                     "width": [128, 256, 64, 64],
                     "batch_norm_layers": ["net.3.2.b1.1",
                                           "net.3.2.b2.4",
                                           "net.3.2.b3.4",
                                           "net.3.2.b4.2"],
                     "next_layers": ["net.3.3.b1.0",
                                     "net.3.3.b2.0",
                                     "net.3.3.b3.0",
                                     "net.3.3.b4.1"]},
    # Inception block 6
    "inception6.1": {"target_conv_layers": ["net.3.3.b2.0"],
                     "width": [144],
                     "batch_norm_layers": ["net.3.3.b2.1"],
                     "next_layers": ["net.3.3.b2.3"]},
    "inception6.2": {"target_conv_layers": ["net.3.3.b3.0"],
                     "width": [32],
                     "batch_norm_layers": ["net.3.3.b3.1"],
                     "next_layers": ["net.3.3.b3.3"]},
    "inception6.3": {"target_conv_layers": ["net.3.3.b1.0",
                                            "net.3.3.b2.3",
                                            "net.3.3.b3.3",
                                            "net.3.3.b4.1"],
                     "width": [112, 288, 64, 64],
                     "batch_norm_layers": ["net.3.3.b1.1",
                                           "net.3.3.b2.4",
                                           "net.3.3.b3.4",
                                           "net.3.3.b4.2"],
                     "next_layers": ["net.3.4.b1.0",
                                     "net.3.4.b2.0",
                                     "net.3.4.b3.0",
                                     "net.3.4.b4.1"]},
    # Inception block 7
    "inception7.1": {"target_conv_layers": ["net.3.4.b2.0"],
                     "width": [160],
                     "batch_norm_layers": ["net.3.4.b2.1"],
                     "next_layers": ["net.3.4.b2.3"]},
    "inception7.2": {"target_conv_layers": ["net.3.4.b3.0"],
                     "width": [32],
                     "batch_norm_layers": ["net.3.4.b3.1"],
                     "next_layers": ["net.3.4.b3.3"]},
    "inception7.3": {"target_conv_layers": ["net.3.4.b1.0",
                                            "net.3.4.b2.3",
                                            "net.3.4.b3.3",
                                            "net.3.4.b4.1"],
                     "width": [256, 320, 128, 128],
                     "batch_norm_layers": ["net.3.4.b1.1",
                                           "net.3.4.b2.4",
                                           "net.3.4.b3.4",
                                           "net.3.4.b4.2"],
                     "next_layers": ["net.4.0.b1.0",
                                     "net.4.0.b2.0",
                                     "net.4.0.b3.0",
                                     "net.4.0.b4.1"]},
    # Inception block 8
    "inception8.1": {"target_conv_layers": ["net.4.0.b2.0"],
                     "width": [160],
                     "batch_norm_layers": ["net.4.0.b2.1"],
                     "next_layers": ["net.4.0.b2.3"]},
    "inception8.2": {"target_conv_layers": ["net.4.0.b3.0"],
                     "width": [32],
                     "batch_norm_layers": ["net.4.0.b3.1"],
                     "next_layers": ["net.4.0.b3.3"]},
    "inception8.3": {"target_conv_layers": ["net.4.0.b1.0",
                                            "net.4.0.b2.3",
                                            "net.4.0.b3.3",
                                            "net.4.0.b4.1"],
                     "width": [256, 320, 128, 128],
                     "batch_norm_layers": ["net.4.0.b1.1",
                                           "net.4.0.b2.4",
                                           "net.4.0.b3.4",
                                           "net.4.0.b4.2"],
                     "next_layers": ["net.4.1.b1.0",
                                     "net.4.1.b2.0",
                                     "net.4.1.b3.0",
                                     "net.4.1.b4.1"]},
    # Inception block 9
    "inception9.1": {"target_conv_layers": ["net.4.1.b2.0"],
                     "width": [192],
                     "batch_norm_layers": ["net.4.1.b2.1"],
                     "next_layers": ["net.4.1.b2.3"]},
    "inception9.2": {"target_conv_layers": ["net.4.1.b3.0"],
                     "width": [48],
                     "batch_norm_layers": ["net.4.1.b3.1"],
                     "next_layers": ["net.4.1.b3.3"]},
    "inception9.3": {"target_conv_layers": ["net.4.1.b1.0",
                                            "net.4.1.b2.3",
                                            "net.4.1.b3.3",
                                            "net.4.1.b4.1"],
                     "width": [384, 384, 128, 128],
                     "batch_norm_layers": ["net.4.1.b1.1",
                                           "net.4.1.b2.4",
                                           "net.4.1.b3.4",
                                           "net.4.1.b4.2"],
                     "next_layers": ["net.5"]},
}
