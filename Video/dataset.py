import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class MultiDataset(Dataset):
    def __init__(self, csv_file, root_dir, nb_class, max_video_length):
        self.df_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.nb_class = nb_class
        self.max_video_length = max_video_length

    def __len__(self):
        return len(self.df_csv)

    def __getitem__(self, idx):
        # Load the video feature (.npy file)
        data_path = os.path.join(self.root_dir, self.df_csv.iloc[idx, 1])
        video_feature = np.load(data_path)  # This should load the stacked vertical and horizontal motiongrams
        
        # Flatten video_feature if necessary
        video_feature_flat = video_feature.reshape(-1)  # Change this depending on the shape of your data
        
        # Truncate or pad the video feature vector to a fixed size
        feature_length = video_feature_flat.shape[0]
        if feature_length > self.max_video_length:
            feat = video_feature_flat[:self.max_video_length]  # Truncate
        else:
            feat = np.pad(video_feature_flat, (0, self.max_video_length - feature_length), 'constant')  # Pad
        
        feat = torch.tensor(feat, dtype=torch.float32)
        
        # Get the label from the dataframe and convert it to a one-hot encoded tensor
        label = self.df_csv.iloc[idx, 2]
        label_one_hot = torch.zeros(self.nb_class)
        label_one_hot[label] = 1

        return feat, label_one_hot