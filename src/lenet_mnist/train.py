"""
Function to train the model for one epoch.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def train(model: nn.Module,
          device: str,
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion: nn.Module,
          epoch: int) -> tuple:
    """Train the model for one epoch

    Args:
        model (Model): The model to train
        device (string): The device to use (cpu or cuda)
        train_loader (Dataloader): The training data loader
        optimizer (Optimizer): The optimizer to use
        criterion (Loss): The classification loss to use
        epoch (int): The current epoch

    Returns:
        double, double: The training loss, the training accuracy
    """
    # Initialize the training loss and accuracy
    train_loss = 0.
    train_correct = 0
    
    # Configure the model for training
    # (good practice, only necessary if the model operates differently for
    # training and validation)
    model.train()
    
    # Add a progress bar
    train_loader_pbar = tqdm(train_loader, unit="batch")
    
    # Loop over the training batches
    for images, labels in train_loader_pbar:
        
        # Print the epoch and training mode
        train_loader_pbar.set_description(f"Epoch {epoch} [train]")
        
        # Move images and labels to GPU (if available)
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero out gradients before each backpropagation pass, to avoid that
        # they accumulate
        optimizer.zero_grad()
        
        # Perform forward pass
        predicted_labels = model(images)
        
        # Compute loss 
        loss = criterion(predicted_labels, labels)
        
        # Print the batch loss next to the progress bar
        train_loader_pbar.set_postfix(batch_loss=loss.item())
        
        # Perform backpropagation (update weights)
        loss.backward()
        
        # Adjust parameters based on gradients
        optimizer.step()
        
        # Accumulate batch loss to average over the epoch
        train_loss += loss.item()
    
        # Get the number of correct predictions
        train_correct += torch.sum(
            torch.argmax(predicted_labels, dim=1) == labels
            ).item()
        
    
    # Compute the losses and accuracies
    train_loss /= len(train_loader)
    train_accuracy = 100*train_correct/len(train_loader.dataset)
    
    return train_loss, train_accuracy
