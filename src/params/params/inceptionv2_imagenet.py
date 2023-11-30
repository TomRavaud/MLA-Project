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
            "momentum": 0.9  # Only used for SGD
            }

# Set the number of classes in the dataset
NB_CLASSES = 1000


####################################
## Images' transforms parameters ##
####################################

# Shape of the images in the ImageNet dataset
IMAGE_SHAPE = (224, 224)
NB_CHANNELS = 3

# Define the mean and std of the dataset
# (pre-computed on the ImageNet dataset)
NORMALIZE_PARAMS = {"mean": (0.485, 0.456, 0.406),
                    "std": (0.229, 0.224, 0.225)}


##################
## Net2WiderNet ##
##################

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
