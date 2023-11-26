"""
This file contains the parameters used to train the Inception-V2 network on
the ImageNet dataset.
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
NB_CLASSES = 1000


####################################
## Images' transforms parameters ##
####################################

#FIXME: Update this shape to match the shape of the images in the ImageNet
# dataset
IMAGE_SHAPE = (32, 32)


#FIXME: Update the values of the normalization parameters
# Define the mean and std of the dataset
# (pre-computed on the ImageNet dataset)
NORMALIZE_PARAMS = {"mean": (0.1307,),
                    "std": (0.3105,)}
