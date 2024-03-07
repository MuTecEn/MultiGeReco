import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from dataset_audio import MultiDataset
from utils.logger import setup_logging
from model import get_model
from config import get_config
import matplotlib.pyplot as plt

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.unsqueeze(1)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.argmax(dim=1)  # Convert one-hot labels to class indices
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.unsqueeze(1)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            labels = labels.argmax(dim=1)  # Convert one-hot labels to class indices
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def main():
    config = get_config()
    device = torch.device(config["device"])

    train_losses = []
    val_losses = []

    train_dataset = MultiDataset(csv_file=config["dataset_path"] + 'train.csv',
                                 root_dir=config["dataset_path"], nb_class=config["n_class"])  
    val_dataset = MultiDataset(csv_file=config["dataset_path"] + 'test.csv',
                               root_dir=config["dataset_path"], nb_class=config["n_class"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    model = get_model(config["model"], config["n_class"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["optimizer"]["lr"], weight_decay=config["optimizer"]["weight_decay"])

    best_val_loss = np.inf
    for epoch in range(config["epochs"]):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Plotting the training and validation loss
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Losses')
    plt.legend()
    plt.savefig('audio_loss.png')
    plt.show()

    # Save the model if validation loss has decreased
    if val_loss < best_val_loss:
        print(f'Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...')
        torch.save(model.state_dict(), config["save_path"])
        best_val_loss = val_loss


if __name__ == '__main__':
    main()