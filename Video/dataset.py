import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MultiDataset(Dataset):
    def __init__(self, csv_file, root_dir, nb_class):
        self.df_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.nb_class = nb_class
        self.transform = transforms.Normalize(mean=[0.4914], std=[0.2023])

    def __len__(self):
        return len(self.df_csv)

    def __getitem__(self, idx):
        data_path = os.path.join(self.root_dir, self.df_csv.iloc[idx, 1])
        features = np.load(data_path)

        # Select the first line of the features
        feat = features[0]  
        feat = np.pad(feat, (0, 1225 - feat.shape[0]), 'constant')
        # Convert the feature array into a PyTorch tensor
        feat = torch.tensor(feat, dtype=torch.float32)

        # One hot encoding for the label
        label = self.df_csv.iloc[idx, 2]
        label_one_hot = torch.zeros(self.nb_class)
        label_one_hot[label] = 1

        return feat, label_one_hot
