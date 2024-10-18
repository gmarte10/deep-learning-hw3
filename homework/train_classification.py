import torch
import torch.utils.tensorboard as tb
import numpy as np
from pathlib import Path
from datetime import datetime
from .models import load_model, save_model
from .datasets.classification_dataset import load_data
from .metrics import AccuracyMetric

def train_classification(
        # Export directory for tensorboard logs and model checkpoints
        exp_dir: str = "logs",
        model_name: str = "classification",
        num_epoch: int = 50,
        # Learning rate for the optimizer
        lr: float = 0.001,
        batch_size: int = 128,
        # Random seed for reproducibility
        seed: int = 2024,
        # Additional keyword arguments to pass to the model (optimizer, decay, etc.)
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

    # Grader uses default kwargs, you can customize them; set model to training mode
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # Load the data; can use SuperTuxDataset from classification dataset module to augment data
    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data=load_data("classification_data/val", shuffle=False)

    # Create loss function and optimizer; can add momentum, weight decay, etc.
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Used to keep track of the x axis in tensorboard plot
    global_step = 0
    # Store the training and validation accuracy
    acc_storage = {"train_accuracy": [], "validation_accuracy": []}
    acc_metric = AccuracyMetric()

    for epoch in range(num_epoch):
        # Clear metrics storage at the start of each epoch
        for key in acc_storage:
            acc_storage[key].clear()

        # Set model to training mode
        model.train()

        # Reset metrics
        acc_metric.reset()

        for img, label in train_data:
            # Put img and label on GPU
            img, label = img.to(device), label.to(device)

            # Predict image label
            pred = model(img)

            # Compute loss value
            loss_val = loss_func(pred, label)

            # Backpropagation
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Compute the training accuracy and store it
            _, predicted = torch.max(pred, 1)
            acc_metric.add(predicted, label)

            global_step += 1

        # Store the training accuracy
        acc_storage["train_accuracy"].append(acc_metric.compute())

        # Disable gradient compution and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            acc_metric.reset()
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                pred = model(img)
                _, predicted = torch.max(pred, 1)
                acc_metric.add(predicted, label)
            acc_storage["validation_accuracy"].append(acc_metric.compute())
        
        # Get the mean training and validation accuracy for the epoch
        epoch_train_acc = torch.as_tensor(acc_storage["train_accuracy"]).mean()
        epoch_val_acc = torch.as_tensor(acc_storage["validation_accuracy"]).mean()

        # Log the training and validation accuracy
        logger.add_scalar("train_accuracy", epoch_train_acc, global_step)
        logger.add_scalar("validation_accuracy", epoch_val_acc, global_step)






