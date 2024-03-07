import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

class Simple1DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)
        
        self.num_flat_features = self._get_conv_output(431)
        
        self.fc1 = nn.Linear(self.num_flat_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _get_conv_output(self, shape):
        
        input = torch.autograd.Variable(torch.rand(1, 1, shape))
        output = self.pool(self.conv1(input))
        output = self.pool(self.conv2(output))
        output = self.pool(self.conv3(output))
        return int(numpy.prod(output.size()))
    
    

    def forward(self, x):
        # Assume x has shape (batch_size, channels, sequence_length)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # Flatten the output for the dense layer
        x = x.view(-1, self.num_flat_features)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_model(model_name, num_classes):
    if model_name == 'Simple1DCNN':
        return Simple1DCNN(num_classes=num_classes)