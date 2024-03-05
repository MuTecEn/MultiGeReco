import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import librosa

class MultiDataset(Dataset):

    def __init__(self, audio_file, root_dir, nb_class):
        self.df_csv = pd.read_csv(audio_file)
        self.root_dir = root_dir
        self.nb_class = nb_class
        
        self.file_names = [f for f in os.listdir(root_dir) if f.endswith(".wav")]

    def __len__(self):
        return len(self.df_csv)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        audio_file_path = os.path.join(self.root_dir, file_name)
        
        # Load audio file
        y, sr = librosa.load(audio_file_path, sr=None)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        
        # Concatenate the features
        features = np.concatenate((mfcc, chromagram, tempogram), axis=0)

        # Pad or truncate to a fixed length (e.g., 1000 frames)
        max_time_frames = 1000
        if features.shape[1] < max_time_frames:
            padding = np.zeros((features.shape[0], max_time_frames - features.shape[1]))
            features = np.concatenate((features, padding), axis=1)
        else:
            features = features[:, :max_time_frames]
        
        # Convert features to PyTorch tensor
        features = torch.tensor(features, dtype=torch.float32)
        
        label = self.df_csv.iloc[idx, 2]
        
        # Create one-hot encoding for the label
        label_one_hot = torch.zeros(self.nb_class)
        label_one_hot[label] = 1

        return features, label_one_hot