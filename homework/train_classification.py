import torch
import argparse
import torch.utils.tensorboard as tb
import numpy as np

from pathlib import Path
from datetime import datetime
from .models import load_model, save_model
from .datasets.classification_dataset import load_data, SuperTuxDataset
from .metrics import AccuracyMetric


def train_classification(
        # Export directory for tensorboard logs and model checkpoints
        exp_dir: str = "logs",
        model_name: str = "classification",
        num_epoch: int = 50,
        # Learning rate for the optimizer
        lr: float = 0.00005,
        batch_size: int = 128,
        # Random seed for reproducibility
        seed: int = 2024,
        # Additional keyword arguments to pass to the model
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
    train_data = load_data("classification_data/train", transform_pipeline = "aug", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data=load_data("classification_data/val", shuffle=False, num_workers=2)

    # AI: Used to figure out what weight decay to use and how to use it
    # Create loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Used to for tracking; keeps track of the x axis in tensorboard plot
    global_step = 0

    # Store the training and validation accuracy
    acc_storage = {"train_accuracy": [], "validation_accuracy": []}

    # Used to create accuracy
    acc_metric = AccuracyMetric()

    for epoch in range(num_epoch):
        # Set model to training mode
        model.train()

        # Reset metrics
        for key in acc_storage:
            acc_storage[key].clear()
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

            # Add metrics
            _, predicted = torch.max(pred, 1)
            acc_metric.add(predicted, label)

            global_step += 1

        # Store the training accuracy
        acc_storage["train_accuracy"].append(acc_metric.compute())
        
        # Disable gradient compution and switch to evaluation mode
        model.eval()
        acc_metric.reset()
        with torch.inference_mode():
            for img, label in val_data:
                # Put img and label on GPU
                img, label = img.to(device), label.to(device)

                # Predict image label
                pred = model(img)

                _, predicted = torch.max(pred, 1)

                # Add metrics
                acc_metric.add(predicted, label)
            
            # Store the validation accuracy
            acc_storage["validation_accuracy"].append(acc_metric.compute())
        
        # AI: Used to figure out why I could not access accuracy and how to parse and convert it properly
        # Parse the training accuracy
        train_acc_list = acc_storage["train_accuracy"]
        train_acc_list2 = []
        for acc in train_acc_list:
            a = acc["accuracy"]
            train_acc_list2.append(torch.tensor(a).float())

        # AI: Used to figure out why I could not access accuracy and how to parse and convert it properly
        # Parse the validation accuracy
        val_acc_list = acc_storage["validation_accuracy"]
        val_acc_list2 = []
        for acc in val_acc_list:
            a = acc["accuracy"]
            val_acc_list2.append(torch.tensor(a).float())

        # Compute the mean of the training and validation accuracy
        epoch_train_acc = torch.as_tensor(train_acc_list2).mean()
        epoch_val_acc = torch.as_tensor(val_acc_list2).mean()

        # Log the training and validation accuracy
        logger.add_scalar("train_accuracy", epoch_train_acc, global_step)
        logger.add_scalar("validation_accuracy", epoch_val_acc, global_step)

        # Print on first, last and every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # Save and overwrite the model in the root directory
    save_model(model)

    # Save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    # Define the arguments for the train_classification function
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2024)

    # Pass all arguments to train_classification
    train_classification(**vars(parser.parse_args()))





