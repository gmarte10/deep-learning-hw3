import torch
import argparse
import torch.utils.tensorboard as tb
import numpy as np
from pathlib import Path
from datetime import datetime
from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric

def train_detection(
        # Export directory for tensorboard logs and model checkpoints
        exp_dir: str = "logs",
        model_name: str = "detector",
        num_epoch: int = 50,
        # Learning rate for the optimizer
        lr: float = 0.001,
        batch_size: int = 16,
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
    train_data = load_data("road_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("road_data/val", shuffle=False, num_workers=2)

    # Create loss function and optimizer; can add momentum, weight decay, etc.
    segmentation_loss = torch.nn.CrossEntropyLoss()
    depth_loss = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Metrics storage
    metrics = {
        "train": {"total_loss": [], "iou": [], "abs_depth_error": []},
        "val": {"total_loss": [], "iou": [], "abs_depth_error": []},
    }

    # Used to keep track of the x axis in tensorboard plot
    global_step = 0
    # Store the training and validation accuracy
    # detection_metric = DetectionMetric()
    # d_metric = {"training":[], "validation": []}
    train_metrics = DetectionMetric()

    for epoch in range(num_epoch):
        # Set model to training mode
        model.train()

        # Reset metrics
        train_metrics.reset()

        for batch in train_data:
            # Put img and label on GPU
            img = batch["image"].to(device)
            depth = batch["depth"].to(device)
            segmentation = batch["track"].to(device)

            optimizer.zero_grad()

            # Predict image label
            segmentation_pred, depth_pred = model(img)

            # Compute loss value
            seg_loss = segmentation_loss(segmentation_pred, segmentation)
            d_loss = depth_loss(depth_pred, depth)

            total_loss = seg_loss + d_loss
            total_loss.backward()
            optimizer.step()

            _, seg_pred = torch.max(segmentation_pred, 1)
            _, depth_pred = torch.max(depth_pred, 1)

            # Add metrics for current batch
            train_metrics.add(seg_pred, segmentation, depth_pred, depth)
        

            global_step += 1

        # Store the training accuracy
        # Compute epoch-wide metrics for training
        train_epoch_metrics = train_metrics.compute()
        metrics["train"]["total_loss"].append(total_loss.item())
        metrics["train"]["iou"].append(train_epoch_metrics["iou"])
        metrics["train"]["abs_depth_error"].append(train_epoch_metrics["abs_depth_error"])

        logger.add_scalar("train/total_loss", total_loss.item(), global_step)
        logger.add_scalar("train/iou", train_epoch_metrics["iou"], global_step)
        logger.add_scalar("train/abs_depth_error", train_epoch_metrics["abs_depth_error"], global_step)
        
        model.eval()
        val_metrics = DetectionMetric()
        # Disable gradient compution and switch to evaluation mode
        with torch.inference_mode():
            for batch in val_data:
                # Put img and label on GPU
                img = batch["image"].to(device)
                depth = batch["depth"].to(device)
                segmentation = batch["track"].to(device)

                # Predict image label
                segmentation_pred, depth_pred = model(img)

                # Compute loss value
                seg_loss = segmentation_loss(segmentation_pred, segmentation)
                d_loss = depth_loss(depth_pred, depth)

                total_loss = seg_loss + d_loss

                _, seg_pred = torch.max(segmentation_pred, 1)

                val_metrics.add(seg_pred, segmentation, depth_pred, depth)
            val_epoch_metrics = val_metrics.compute()
            metrics["val"]["total_loss"].append(total_loss.item())
            metrics["val"]["iou"].append(val_epoch_metrics["iou"])
            metrics["val"]["abs_depth_error"].append(val_epoch_metrics["abs_depth_error"])

            logger.add_scalar("val/total_loss", total_loss.item(), global_step)
            logger.add_scalar("val/iou", val_epoch_metrics["iou"], global_step)
            logger.add_scalar("val/abs_depth_error", val_epoch_metrics["abs_depth_error"], global_step)
        
        

        # Print on first, last and every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}: Train IoU={train_epoch_metrics['iou']:.4f}, Val IoU={val_epoch_metrics['iou']:.4f}"
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
    train_detection(**vars(parser.parse_args()))





