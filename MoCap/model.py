import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from dataset import MultiDataset
from utils.logger import setup_logging
from model import get_model
from config import get_config
import matplotlib.pyplot as plt

def train(model, train_loader, criterion, optimizer, device):
    model.train()  # Sets the model to training mode
    running_loss = 0.0  # Variable to store the running loss
    for batch_idx, (inputs, labels) in enumerate(train_loader):  # Iterate over the training data
        inputs = inputs.unsqueeze(1)  # Add a single channel dimension to the input data
        inputs, labels = inputs.to(device), labels.to(device)  # Move the input data and labels to the specified device (e.g., CPU or GPU)

        optimizer.zero_grad()  # Clear the gradients of all optimized tensors
        outputs = model(inputs)  # Forward pass: compute the predicted outputs by passing inputs to the model
        labels = labels.argmax(dim=1)  # Convert one-hot encoded labels to class indices
        loss = criterion(outputs, labels)  # Compute the loss between the predicted outputs and the actual labels
        loss.backward()  # Backward pass: compute the gradients of the loss with respect to model parameters
        optimizer.step()  # Update the model's parameters based on the computed gradients

        running_loss += loss.item()  # Add the current batch's loss to the running loss
    return running_loss / len(train_loader)  # Calculate and return the average loss over all training batches


def validate(model, val_loader, criterion, device):
    model.eval()  
    val_loss = 0.0  
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():  
        for inputs, labels in val_loader:  
            inputs = inputs.unsqueeze(1)  
            inputs, labels = inputs.to(device), labels.to(device)  

            outputs = model(inputs) 
            labels = labels.argmax(dim=1)  
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Collect true labels and predicted labels
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(outputs.argmax(dim=1).cpu().numpy())
    
    return val_loss / len(val_loader), true_labels, predicted_labels

def calculate_accuracy(true_labels, predicted_labels):
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    total = len(true_labels)
    accuracy = correct / total
    return accuracy

def main():

    config = get_config()  # Load configuration settings
    device = torch.device(config["device"])  # Set the device for model training (e.g., CPU or GPU)

    train_losses = []  # Initialize a list to store the training losses
    val_losses = []  # Initialize a list to store the validation losses

    # Load the training and validation datasets
    train_dataset = MultiDataset(csv_file=config["dataset_path"] + 'train.csv',
                                 root_dir=config["dataset_path"], nb_class=config["n_class"])
    val_dataset = MultiDataset(csv_file=config["dataset_path"] + 'test.csv',
                               root_dir=config["dataset_path"], nb_class=config["n_class"])

    # Create data loaders for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    # Get the model and move it to the specified device
    model = get_model(config["model"], config["n_class"]).to(device)

    # Set the loss function and optimizer for training the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"])

    best_val_loss = np.inf  # Initialize the best validation loss with infinity
    for epoch in range(config["epochs"]):  # Iterate over the specified number of epochs
        # Train the model and get the training loss for the current epoch
        train_loss = train(model, train_loader, criterion, optimizer, device)
        
        # Validate the model and get the validation loss for the current epoch
        val_loss, true_labels, predicted_labels = validate(model, val_loader, criterion, device)
        accuracy = calculate_accuracy(true_labels, predicted_labels)

        # Append the training and validation losses to their respective lists 
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print the training and validation losses for the current epoch
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')


    # Plot the training and validation losses
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Losses')
    plt.legend()
    plt.savefig('mocap_loss.png')  # Save the plot as an image
    plt.show()  # Display the plot

    # Save the model if the validation loss has decreased
    if val_loss < best_val_loss:
        print(f'Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...')
        torch.save(model.state_dict(), '/itf-fi-ml/shared/users/annammc/Anna/save/mocap_model.pth') # Save the model's state dictionary
        best_val_loss = val_loss  # Update the best validation loss

if __name__ == '__main__':
    main()  # Execute the main function if the script is being run as the main program