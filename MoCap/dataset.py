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

        # resultant vector calculation
        L_hand = np.sqrt(np.square(features[1,:]) + np.square(features[2,:]) + np.square(features[3,:]))
        R_hand = np.sqrt(np.square(features[5,:]) + np.square(features[6,:]) + np.square(features[7,:]))
        
        # Max mocap feature length: 4897
        feat = np.concatenate((L_hand, R_hand), axis=None)                
        feat = np.pad(feat, (0, 4897*2 - feat.shape[0]), 'constant')
        feat = torch.tensor(feat, dtype=torch.float32)
        
        label = self.df_csv.iloc[idx, 2]
         # Create one-hot encoding for the label
        label_one_hot = torch.zeros(self.nb_class)
        label_one_hot[label] = 1

        return feat, label_one_hot