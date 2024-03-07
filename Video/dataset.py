import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class MultiDataset(Dataset):
    def __init__(self, csv_file, root_dir, nb_class):
        self.df_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.nb_class = nb_class

    def __len__(self):
        return len(self.df_csv)

    def __getitem__(self, idx):
        data_path = os.path.join(self.root_dir, self.df_csv.iloc[idx, 1])
        features = np.load(data_path)

        # Select the motiongram
        feat = features[:, :, 0]  # Select the vertical motiongram
        feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

        label = self.df_csv.iloc[idx, 2]
         # Create one-hot encoding for the label
        label_one_hot = torch.zeros(self.nb_class)
        label_one_hot[label] = 1

        return feat, label