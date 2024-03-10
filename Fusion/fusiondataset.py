import os
import torch
import numpy as np
import pandas as pd

class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, fused_representations):
        self.fused_representations = fused_representations

    def __len__(self):
        return len(self.fused_representations)

    def __getitem__(self, idx):
        representation_label = self.fused_representations[idx]
        input_data = representation_label[:-1]  # Extract input data from the fused representation
        input_data = input_data.to(torch.float32)
        label = representation_label[-1]  # Extract the label from the fused representation
        return input_data, label


