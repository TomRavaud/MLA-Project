"""
Main script to train, evaluate and test a model on the CIFAR-10 dataset.
"""

# Import libraries
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from datetime import datetime
import os


# Import custom modules and packages
import params.inceptionv2_cifar
import utils.log
import utils.train
import utils.test
import utils.validate
from models.inceptionv2 import GoogleNetBN
import net2net.net2net_wider


def full_training(learning_params: dict,
                  model: torch.nn.Module,
                  generate_logs: bool=True) -> torch.nn.Module:
    """Train, evaluate and test a model on the CIFAR-10 dataset.

    Args:
        learning_params (dict): Parameters used for the training
        model (torch.nn.Module): Model to train
        generate_logs (bool, optional): Whether to generate logs or not.
        Defaults to True.

    Returns:
        torch.nn.Module: Trained model
    """

    # Compose several transforms together to be applied to data
    # (Note that transforms are not applied yet)
    transform = transforms.Compose([
        # Modify the size of the images
        transforms.Resize(params.inceptionv2_cifar.IMAGE_SHAPE),

        # Convert a PIL Image or numpy.ndarray to tensor
        transforms.ToTensor(),

        # Normalize a tensor image with pre-computed mean and standard
        # deviation (based on the data used to train the model(s))
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


    # As CIFAR-10 does not provide a validation dataset, we will split the
    # train dataset into a train and a validation dataset

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
        batch_size = learning_params['batch_size'],
        shuffle = True,
        num_workers=10,  # Asynchronous data loading and augmentation
        pin_memory=True,  # Increase the transferring speed to the GPU
    )

    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = learning_params['batch_size'],
        shuffle = False,  # SequentialSampler
        num_workers=10,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=learning_params["batch_size"],
        shuffle=True,
        num_workers=10,
        pin_memory=True,
    )

    # Use a GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Move the model to the device
    model.to(device)

    # Define the loss function (combines nn.LogSoftmax() and nn.NLLLoss())
    criterion = torch.nn.CrossEntropyLoss()

    # Set the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_params["learning_rate"],
        weight_decay=learning_params["weight_decay"])

    # Create tensors to store the loss and accuracy values
    loss_values = torch.zeros(2, learning_params["nb_epochs"])
    accuracy_values = torch.zeros(2, learning_params["nb_epochs"])


    # Loop over the epochs
    for epoch in range(learning_params["nb_epochs"]):

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

    # Generate logs
    if generate_logs:
        
        # Test the model
        _, test_accuracy = utils.test.test(model,
                                                   device,
                                                   test_loader,
                                                   criterion)

        # Get the learning parameters table
        params_table = utils.log.parameters_table(
            dataset=params.inceptionv2_cifar.DATASET_NAME,
            learning_params=learning_params)

        # Set the path to the results directory
        results_directory = params.inceptionv2_cifar.PATH +\
            f"/logs/{params.inceptionv2_cifar.STUDY_NAME}/" +\
                datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # Generate the log directory
        utils.log.generate_log(
            results_directory=results_directory,
            test_accuracy=test_accuracy,
            parameters_table=params_table,
            model=model,
            accuracy_values=accuracy_values)
    
    return model


# Make the logs directory if it does not exist
if not os.path.exists(params.inceptionv2_cifar.PATH +\
                      f"/logs/{params.inceptionv2_cifar.STUDY_NAME}"):
    os.makedirs(params.inceptionv2_cifar.PATH +\
                f"/logs/{params.inceptionv2_cifar.STUDY_NAME}")

# Loading the training parameters
LEARNING_PARAMS = params.inceptionv2_cifar.LEARNING


for _ in range(params.inceptionv2_cifar.NB_SEEDS):
        
    # Instantiate a teacher model
    teacher_model = GoogleNetBN(
        nb_classes=params.inceptionv2_cifar.NB_CLASSES,
        inception_factor=params.inceptionv2_cifar.INCEPTION_FACTOR)
    
    # Get the learning rate of the teacher model and update the dictionary
    # of learning parameters
    LEARNING_PARAMS_TEACHER = LEARNING_PARAMS.copy()
    
    if params.inceptionv2_cifar.INCEPTION_FACTOR == 1:
        LEARNING_PARAMS_TEACHER.update(
            {"learning_rate": params.inceptionv2_cifar.LR_RANDOM_INITIALIZATION})
    else:
        LEARNING_PARAMS_TEACHER.update(
            {"learning_rate": params.inceptionv2_cifar.LR_TEACHER})
    
    # Train the teacher model and get the trained model
    teacher_model = full_training(LEARNING_PARAMS_TEACHER,
                                  teacher_model)
    
    
    # Train the student model
    if params.inceptionv2_cifar.TRAIN_STUDENT:
        
        # Instantiate a Net2Net object from a (pre-trained) model
        my_net2net = net2net.net2net_wider.Net2Net(
            teacher_network=teacher_model)

        # Apply the Net2Net widening operations and get the student network
        my_net2net.net2wider(
            wider_operations=params.inceptionv2_cifar.wider_operations,
            sigma=params.inceptionv2_cifar.SIGMA,
            random_pad=params.inceptionv2_cifar.RANDOM_PAD)
        student_model = my_net2net.get_student_network()
        
        # Get the learning rate of the student model and update the dictionary
        # of learning parameters
        LEARNING_PARAMS_STUDENT = LEARNING_PARAMS.copy()
        
        if params.inceptionv2_cifar.RANDOM_PAD:
            LEARNING_PARAMS_STUDENT.update(
                {"learning_rate": params.inceptionv2_cifar.LR_RANDOM_PAD})
        else:
            LEARNING_PARAMS_STUDENT.update(
                {"learning_rate": params.inceptionv2_cifar.LR_NET2WIDERNET})
        
        # Train the student model
        full_training(LEARNING_PARAMS_STUDENT, student_model)
