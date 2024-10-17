import torch
import torch.utils.tensorboard as tb
import numpy as np
from pathlib import Path
from datetime import datetime

def train_classification(
        exp_dir: str = "logs",
        model_name: str = "classification",
        num_epoch: int = 50,
        lr: float = 0.001,
        batch_size: int = 128,
        seed: int = 2024,
        **kwargs,
):
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("Cuda is not available. Training on CPU")
        device = torch.device("cpu")
    
    # Set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)