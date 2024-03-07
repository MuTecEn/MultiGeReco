import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple2DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple2DCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        input_shape = (2, 240, 320)  # Input shape is (height, width, channels)
        self.num_flat_features = self._get_conv_output(input_shape)  # Calculate the number of flat features
        
        print("Creating fc1 layer")
        self.fc1 = nn.Linear(self.num_flat_features, 128)  # Add the missing fc1 layer
        self.fc2 = nn.Linear(128, num_classes)  # Use the provided num_classes argument here

    def _get_conv_output(self, shape):
        dummy_input = torch.zeros(1, *shape)  # Create a dummy input tensor with the specified shape
        output_features = self.forward(dummy_input)  # Pass the dummy input through the model's convolutional layers
        num_flat_features = output_features.reshape(1, -1).shape[1]  # Calculate the number of flat features
        return num_flat_features
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten the tensor before passing it through the fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_model(model_name, num_classes):
    print("Creating Simple2DCNN instance")
    if model_name.lower() == "simple2dcnn":  # Using lower() to compare case-insensitively
        return Simple2DCNN(num_classes)
    else:
        raise ValueError("Invalid model name: " + model_name)