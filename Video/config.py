import torch

def get_config():
    config = {
        "dataset_path": "/itf-fi-ml/shared/users/annammc/Anna/video/",
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
        "device": "cuda:1" if torch.cuda.is_available() else "cpu",
        "model": "Simple2DCNN",
        "save_path": "./save",
        # "save_path": "/itf-fi-ml/shared/users/annammc/Anna/save",
        "log_step": 100,
        "val_step": 10,
    }
    return config
