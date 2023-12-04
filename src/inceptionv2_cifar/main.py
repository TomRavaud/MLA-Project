# Import libraries
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime


# Import custom modules and packages
import params.inceptionv2_cifar
import utils.log
import utils.train
import utils.test
import utils.validate
from inceptionv2 import GoogleNetBN


# Loading the training parameters
LEARNING_PARAMS = params.inceptionv2_cifar.LEARNING


# Compose several transforms together to be applied to data
# (Note that transforms are not applied yet)
transform = transforms.Compose([
    # Modify the size of the images
    transforms.Resize(params.inceptionv2_cifar.IMAGE_SHAPE),
    
    # Convert a PIL Image or numpy.ndarray to tensor
    transforms.ToTensor(),
    
    # Normalize a tensor image with pre-computed mean and standard deviation
    # (based on the data used to train the model(s))
    # (be careful, it only works on torch.*Tensor)
    transforms.Normalize(**params.inceptionv2_cifar.NORMALIZE_PARAMS),
])


# Load the train dataset
train_dataset = torchvision.datasets.CIFAR10(
    root=params.inceptionv2_cifar.PATH,
    train=True,
    transform=transform,
    download=True,
)

# Load the test dataset
test_dataset=torchvision.datasets.CIFAR10(
    root=params.inceptionv2_cifar.PATH,
    train=False,
    transform=transform,
    download=True,
)


# As CIFAR-10 does not provide a validation dataset, we will split the train
# dataset into a train and a validation dataset

# Start by loading the train dataset, with the same transform as the
# test dataset
val_dataset = torchvision.datasets.CIFAR10(
    root=params.inceptionv2_cifar.PATH,
    train=True,
    transform=transform,
)

# Set the train dataset size as a percentage of the original train dataset
train_size = (len(train_dataset) - len(test_dataset))/len(train_dataset)

# Splits train data indices into train and validation data indices
train_indices, val_indices = train_test_split(range(len(train_dataset)),
                                              train_size=train_size)

# Extract the corresponding subsets of the train dataset
train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
val_dataset = torch.utils.data.Subset(val_dataset, val_indices)


# Combine a dataset and a sampler, and provide an iterable over the dataset
# (setting shuffle argument to True calls a RandomSampler, and avoids to
# have to create a Sampler object)
train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = LEARNING_PARAMS['batch_size'],
    shuffle = True,
    num_workers=20,  # Asynchronous data loading and augmentation
    pin_memory=True,  # Increase the transferring speed of the data to the GPU
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = LEARNING_PARAMS['batch_size'],
    shuffle = False,  # SequentialSampler
    num_workers=20,
    pin_memory=True,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=LEARNING_PARAMS["batch_size"],
    shuffle=True,
    num_workers=20,
    pin_memory=True,
)


# Use a GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")


# Instantiate a model
model = GoogleNetBN(
    nb_classes=params.inceptionv2_cifar.NB_CLASSES,
    inception_factor=params.inceptionv2_cifar.INCEPTION_FACTOR)\
        .to(device=device)

# Define the loss function (combines nn.LogSoftmax() and nn.NLLLoss())
criterion = torch.nn.CrossEntropyLoss()

# Set the optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_PARAMS["learning_rate"],
    weight_decay=LEARNING_PARAMS["weight_decay"])


# Create tensors to store the loss and accuracy values
loss_values = torch.zeros(2, LEARNING_PARAMS["nb_epochs"])
accuracy_values = torch.zeros(2, LEARNING_PARAMS["nb_epochs"])


# Loop over the epochs
for epoch in range(LEARNING_PARAMS["nb_epochs"]):
    
    # Training
    train_loss, train_accuracy = utils.train.train(model,
                                                   device,
                                                   train_loader,
                                                   optimizer,
                                                   criterion,
                                                   epoch)
    
    # Validation
    val_loss, val_accuracy = utils.validate.validate(model,
                                                     device,
                                                     val_loader,
                                                     criterion,
                                                     epoch) 
    
    
    print("Train accuracy: ", train_accuracy)
    print("Validation accuracy: ", val_accuracy)
    
    # Store the computed losses
    loss_values[0, epoch] = train_loss
    loss_values[1, epoch] = val_loss
    # Store the computed accuracies
    accuracy_values[0, epoch] = train_accuracy
    accuracy_values[1, epoch] = val_accuracy


# Test the model
test_loss, test_accuracy = utils.test.test(model,
                                           device,
                                           test_loader,
                                           criterion)
print("Test accuracy: ", test_accuracy)

# Get the learning parameters table
params_table = utils.log.parameters_table(
    dataset=params.inceptionv2_cifar.DATASET_NAME,
    learning_params=LEARNING_PARAMS)

# Set the path to the results directory
results_directory = params.inceptionv2_cifar.PATH + "logs/" +\
    datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                        
# Generate the log directory
utils.log.generate_log(
    results_directory=results_directory,
    test_accuracy=test_accuracy,
    parameters_table=params_table,
    model=model,
    accuracy_values=accuracy_values)
