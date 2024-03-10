import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)  # Add batch normalization after conv1
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)  # Add batch normalization after conv2
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)  # Add batch normalization after conv3

        self.num_flat_features = self._get_conv_output(431)

        self.fc1 = nn.Linear(self.num_flat_features, 256)
        self.bn4 = nn.BatchNorm1d(256)  # Add batch normalization after fc1
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)  # Add batch normalization after fc2
        self.fc3 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)  # Add batch normalization after fc3
        self.fc4 = nn.Linear(64, 32)
        self.bn7 = nn.BatchNorm1d(32)  # Add batch normalization after fc4
        self.fc5 = nn.Linear(32, num_classes)

    def _get_conv_output(self, shape):
        input = torch.autograd.Variable(torch.rand(1, 1, shape))
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        return int(numpy.prod(output.size()))

    def forward(self, x):
        # Assume x has shape (batch_size, channels, sequence_length)
        x = F.relu(self.bn1(self.conv1(x)))  # Apply batch normalization after conv1
        x = F.relu(self.bn2(self.conv2(x)))  # Apply batch normalization after conv2
        x = F.relu(self.bn3(self.conv3(x)))  # Apply batch normalization after conv3
        # Flatten the output for the dense layer
        x = x.view(-1, self.num_flat_features)
        x = F.relu(self.bn4(self.fc1(x)))  # Apply batch normalization after fc1
        x = F.relu(self.bn5(self.fc2(x)))  # Apply batch normalization after fc2
        x = F.relu(self.bn6(self.fc3(x)))  # Apply batch normalization after fc3
        x = F.relu(self.bn7(self.fc4(x)))  # Apply batch normalization after fc4
        x = self.fc5(x)
        return x

def get_model(model_name, num_classes):
    if model_name == 'AudioCNN':
        return AudioCNN(num_classes=num_classes)