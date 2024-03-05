import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2
from sklearn.model_selection import KFold

class MultiDataset(Dataset):

    def __init__(self, video_file, root_dir, nb_class, num_folds=5):
        self.df_csv = pd.read_csv(video_file)
        self.root_dir = root_dir
        self.nb_class = nb_class
        self.num_folds = num_folds

        self.file_names = [f for f in os.listdir(root_dir) if f.endswith(".mov")]

        # Perform K-Fold cross-validation
        self.kf = KFold(n_splits=num_folds, shuffle=True)

    def __len__(self):
        return len(self.df_csv) * self.num_folds

    def __getitem__(self, fold_idx):
        # Get the train and test indices for this fold
        train_indices, test_indices = list(self.kf.split(self.df_csv))[fold_idx]

        # Get the file names for the train and test sets
        train_file_names = [self.file_names[i] for i in train_indices]
        test_file_names = [self.file_names[i] for i in test_indices]

        video_frames_list, features_list, labels_list = [], [], []

        for file_name in train_file_names:
            video_file_path = os.path.join(self.root_dir, file_name)
            video_capture = cv2.VideoCapture(video_file_path)

            # Extract video frames - modify the logic for video frame extraction
            frames = []
            success, image = video_capture.read()
            while success:
                frames.append(image)
                success, image = video_capture.read()

            # Extract features from video frames - add your specific video feature extraction here
            # Example: Get the mean pixel value for each frame
            features = [np.mean(frame) for frame in frames]  # Replace with actual video feature extraction logic

            # Pad or truncate to a fixed length (e.g., 1000 frames)
            max_frames = 1000
            if len(features) < max_frames:
                padding = np.zeros(max_frames - len(features))
                features = np.concatenate((features, padding))
            else:
                features = features[:max_frames]

            # Convert features to PyTorch tensor
            features = torch.tensor(features, dtype=torch.float32)
            
            # Get label from DataFrame
            label = self.df_csv.loc[self.df_csv['filename'] == file_name, 'target'].iloc[0]

            # Create one-hot encoding for the label
            label_one_hot = torch.zeros(self.nb_class)
            label_one_hot[label] = 1

            video_capture.release()

            video_frames_list.append(frames)
            features_list.append(features)
            labels_list.append(label_one_hot)

        return video_frames_list, features_list, labels_list