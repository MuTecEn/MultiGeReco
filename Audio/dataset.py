import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MultiDataset(Dataset):
    def __init__(self, csv_file, root_dir, nb_class, max_length):
        self.df_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.nb_class = nb_class
        self.max_length = max_length  # This would be the maximum sequence length
        
    def __len__(self):
        return len(self.df_csv)

    def __getitem__(self, idx):
        data_path = os.path.join(self.root_dir, self.df_csv.iloc[idx, 1])
        features = np.load(data_path)

        # Assuming the features are already a 2D numpy array of shape (feature_size, time_steps)
        # We will flatten the features to form a 1D array
        flattened_features = features.flatten()    

        # Then we will pad or truncate to the fixed length `self.max_length`
        if flattened_features.shape[0] < self.max_length:
            # Pad the feature to the max_length if it's short
            feature_padded = np.pad(flattened_features, (0, self.max_length - flattened_features.shape[0]), 'constant')
        else:
            # Truncate the feature to max_length if it's long
            feature_padded = flattened_features[:self.max_length]
        
        # Convert the numpy array to a PyTorch tensor
        feat = torch.tensor(feature_padded, dtype=torch.float32)

        # Get the label for the current data point
        label = self.df_csv.iloc[idx, 2]
        # Create one-hot encoding for the label
        label_one_hot = torch.zeros(self.nb_class)
        label_one_hot[label] = 1

        return feat, label_one_hot