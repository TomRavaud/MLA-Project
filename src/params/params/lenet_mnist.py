"""
This file contains the parameters used to train the LeNet network on the MNIST
dataset.
"""

################################
## Parameters to train the NN ##
################################

# Define splits size
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Learning parameters
LEARNING = {"batch_size": 32,
            "nb_epochs": 10,
            "learning_rate": 0.005,
            "weight_decay": 0.0001,
            "momentum": 0.9}


####################################
## Images' transforms parameters ##
####################################

IMAGE_SHAPE = (32, 32)

# Define the mean and std of the dataset
# (pre-computed on the MNIST dataset)
NORMALIZE_PARAMS = {"mean": (0.1307,),
                    "std": (0.3105,)}
