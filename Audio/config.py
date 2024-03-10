import torch

def get_config():
    config = {
        "dataset_path": "/path/to/your/audio/folder/",
        "n_class": 5,
        "batch_size": 32,
        "optimizer": {
            "type": "RMSprop",
            "lr": 1e-3,
            "weight_decay": 10 ** -4
        },
        "scheduler": {
            "step_size": 20,
            "gamma": 0.5
        },
        "epochs": 100,
        "n_folds": 5,
        "device": "cpu",
        "model": "AudioCNN",
        "save_path": "./save",
        "log_step": 100,
        "val_step": 10,
    }
    return config
