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
        data_path =os.path.join(self.root_dir, self.df_csv.iloc[idx, 1])
        features = np.load(data_path)

        # for the np.square get the 1st row for rms or 2nd for scontrast (idx[0 for rms])

        feat = features[0]
        feat = np.pad(feat, (0, 431 - feat.shape[0]), 'constant')
        feat = torch.tensor(feat, dtype=torch.float32)

        label = self.df_csv.iloc[idx, 2]
         # Create one-hot encoding for the label
        label_one_hot = torch.zeros(self.nb_class)
        label_one_hot[label] = 1

        return feat, label_one_hot