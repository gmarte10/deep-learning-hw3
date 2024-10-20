import torch
import argparse
import torch.utils.tensorboard as tb
import numpy as np
from pathlib import Path
from datetime import datetime
from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric

def train_classification(
        # Export directory for tensorboard logs and model checkpoints
        exp_dir: str = "logs",
        model_name: str = "detector",
        num_epoch: int = 50,
        # Learning rate for the optimizer
        lr: float = 0.001,
        batch_size: int = 96,
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
    train_data = load_data("road_data/train", transform_pipeline = "aug", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data=load_data("road_data/val", shuffle=False, num_workers=2)

    # Create loss function and optimizer; can add momentum, weight decay, etc.
    segmentation_loss = torch.nn.CrossEntropyLoss()
    depth_loss = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Metrics storage
    metrics = {
        "train": {"segmentation_loss": [], "depth_loss": [], "iou": [], "abs_depth_error": []},
        "val": {"segmentation_loss": [], "depth_loss": [], "iou": [], "abs_depth_error": []},
    }

    # Used to keep track of the x axis in tensorboard plot
    global_step = 0
    # Store the training and validation accuracy
    # detection_metric = DetectionMetric()
    # d_metric = {"training":[], "validation": []}

    for epoch in range(num_epoch):
        # Set model to training mode
        model.train()
        train_metrics = DetectionMetric()


        # Reset metrics
        train_metrics.reset()

        for batch in train_data:
            # Put img and label on GPU
            img = batch["image"].to(device)
            depth = batch["depth"].to(device)
            track = batch["track"].to(device)

            optimizer.zero_grad()

            # Predict image label
            segmentation_pred, depth_pred = model(img)

            # Compute loss value
            seg_loss = segmentation_loss(segmentation_pred, track)
            d_loss = depth_loss(depth_pred, depth)

            total_loss = seg_loss + d_loss
            total_loss.backward()
            optimizer.step()

            _, seg_pred = torch.max(segmentation_pred, 1)
            _, depth_pred = torch.max(depth_pred, 1)
            train_metrics.add(seg_pred, track, depth_pred, depth)

            # Add metrics for current batch
            train_metrics.add(seg_pred, track, depth_pred, depth)
        

            global_step += 1

        # Store the training accuracy
        # Compute epoch-wide metrics for training
        train_epoch_metrics = train_metrics.compute()
        metrics["train"]["segmentation_loss"].append(seg_loss.item())
        metrics["train"]["depth_loss"].append(d_loss.item())
        metrics["train"]["iou"].append(train_epoch_metrics["iou"])
        metrics["train"]["abs_depth_error"].append(train_epoch_metrics["abs_depth_error"])
        
        model.eval()
        d_metric.reset()
        # Disable gradient compution and switch to evaluation mode
        with torch.inference_mode():
            for batch in val_data:
                # Put img and label on GPU
                img = batch["image"].to(device)
                depth = batch["depth"].to(device)
                track = batch["track"].to(device)

                # Predict image label
                segmentation_pred, depth_pred = model(img)

                # Compute loss value
                seg_loss = segmentation_loss(segmentation_pred, track)
                d_loss = depth_loss(depth_pred, depth)

                total_loss = seg_loss + d_loss

                _, seg_pred = torch.max(segmentation_pred, 1)

                detection_metric.add(seg_pred, track, depth_pred, depth)
            d_metric["validation"].append(d_metric.compute())
        
        # Get the mean training and validation accuracy for the epoch
        # train_acc_list = [torch.tensor(acc).float() for acc in acc_storage["train_accuracy"]]
        # val_acc_list = [torch.tensor(acc).float() for acc in acc_storage["validation_accuracy"]]

        train_abs_list = d_metric["training"]
        train_abs = []
        train_tp = []
        for abs in train_abs_list:
            a = abs["abs_depth_error"]
            a2 = abs["tp_depth_error"]
            train_abs.append(torch.tensor(a).float())
            train_tp.append(torch.tensor(a2).float())

        val_abs_list = d_metric["validation"]
        val_abs = []
        val_tp = []
        for abs in val_abs_list:
            a = abs["abs_depth_error"]
            a2 = abs["tp_depth_error"]
            val_abs.append(torch.tensor(a).float())
            val_tp.append(torch.tensor(a2).float())

        epoch_train_abs = torch.as_tensor(train_abs).mean()
        epoch_train_tp = torch.as_tensor(train_tp).mean()
        epoch_val_abs = torch.as_tensor(val_abs).mean()
        epoch_val_tp = torch.as_tensor(val_tp).mean()

        # Log the training and validation accuracy
        logger.add_scalar("train_abs", epoch_train_abs, global_step)
        logger.add_scalar("train_tp", epoch_train_tp, global_step)
        logger.add_scalar("val_abs", epoch_val_abs, global_step)
        logger.add_scalar("val_tp", epoch_val_tp, global_step)

        # Print on first, last and every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_abs:.4f} "
                f"val_acc={epoch_train_tp:.4f}"
                f"train_acc={epoch_val_abs:.4f} "
                f"val_acc={epoch_val_tp:.4f}"
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





