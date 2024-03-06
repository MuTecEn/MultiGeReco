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

    # Calculate the number of columns and determine the split points
        num_columns = features.shape[0]
        split1 = num_columns // 3
        split2 = 2 * (num_columns // 3)

        # Divide the audio_features into mfcc, chromagram, and tempogram
        mfcc = features[:split1]
        chromagram = features[split1:split2]
        tempogram = features[split2:]

         # Max audio npy length = 148994610
        feat = np.concatenate((mfcc, chromagram, tempogram), axis=None)                
        feat = np.pad(feat, (0, 148994610 - feat.shape[0]), 'constant')
        feat = torch.tensor(feat, dtype=torch.float32)

        # Get the label for the current data point
        label = self.df_csv.iloc[idx, 2]
        # Create one-hot encoding for the label
        label_one_hot = torch.zeros(self.nb_class)
        label_one_hot[label] = 1

        return feat, label_one_hot