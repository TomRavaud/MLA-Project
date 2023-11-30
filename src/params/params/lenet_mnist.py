"""
This file contains the parameters used to train the LeNet network on the MNIST
dataset.
"""

################################
## Parameters to train the NN ##
################################

# Learning parameters
LEARNING = {"batch_size": 64,
            "nb_epochs": 4,
            "learning_rate": 0.005,
            "weight_decay": 0.0001,
            "momentum": 0.9  # Used only for SGD
            }

# Set the number of classes in the dataset
NB_CLASSES = 10


####################################
## Images' transforms parameters ##
####################################

IMAGE_SHAPE = (32, 32)
NB_CHANNELS = 1

# Define the mean and std of the dataset
# (pre-computed on the MNIST dataset)
NORMALIZE_PARAMS = {"mean": (0.1307,),
                    "std": (0.3105,)}
