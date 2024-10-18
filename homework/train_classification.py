import torch
import argparse
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
        # train_acc_list = [torch.tensor(acc).float() for acc in acc_storage["train_accuracy"]]
        # val_acc_list = [torch.tensor(acc).float() for acc in acc_storage["validation_accuracy"]]

        train_acc_list = acc_storage["train_accuracy"]
        train_acc_list2 = []
        for acc in train_acc_list:
            a = acc["accuracy"]
            train_acc_list2.append(torch.tensor(a).float())

        val_acc_list = acc_storage["validation_accuracy"]
        val_acc_list2 = []
        for acc in val_acc_list:
            a = acc["accuracy"]
            val_acc_list2.append(torch.tensor(a).float())

        epoch_train_acc = torch.as_tensor(train_acc_list2).mean()
        epoch_val_acc = torch.as_tensor(val_acc_list2).mean()

        # Log the training and validation accuracy
        logger.add_scalar("train_accuracy", epoch_train_acc, global_step)
        logger.add_scalar("validation_accuracy", epoch_val_acc, global_step)

        # Print on first, last and every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d } / {num_epoch}: "
                f"train_acc = {epoch_train_acc:.4f}"
                f"val_acc = {epoch_val_acc:.4f}"
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





