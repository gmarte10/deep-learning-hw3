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
        lr: float = 0.0001,
        batch_size: int = 32,
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

    # Load the data; can use RoadDataSet from road dataset module to augment data
    train_data = load_data("road_data/train", shuffle=True, transform_pipeline = "aug", batch_size=batch_size, num_workers=2)
    val_data = load_data("road_data/val", shuffle=False, num_workers=2)

    # Create loss functions and optimizer
    segmentation_loss = torch.nn.CrossEntropyLoss()
    depth_loss = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # AI: Simplified the way I was storing metrics by making one dictionary instead of multiple
    # Metrics storage
    metrics = {
        "train": {"total_loss": [], "iou": [], "abs_depth_error": [], "tp_depth_error":[], "total_seg_loss": [], "total_depth_loss": []},
        "val": {"total_loss": [], "iou": [], "abs_depth_error": [], "tp_depth_error":[], "total_seg_loss": [], "total_depth_loss": []},
    }

    # Used to for tracking; keeps track of the x axis in tensorboard plot
    global_step = 0
    
    # Computes iou, abs_depth_error, tp_depth_error
    train_metrics = DetectionMetric()
    val_metrics = DetectionMetric()

    for epoch in range(num_epoch):
        # Set model to training mode
        model.train()

        # Reset metrics
        train_metrics.reset()
        total_train_loss = 0
        total_seg_loss = 0
        seg_len = 0
        total_depth_loss = 0
        depth_len = 0

        for batch in train_data:
            # Put img depth and segmentation data on GPU
            img = batch["image"].to(device)
            depth = batch["depth"].to(device)
            segmentation = batch["track"].to(device)

            optimizer.zero_grad()

            # Predict depth and segmentation
            segmentation_pred, depth_pred = model(img)

            # Compute loss value
            seg_loss = segmentation_loss(segmentation_pred, segmentation)
            d_loss = depth_loss(depth_pred, depth)

            # AI: Showed me how to correctly combine the losses
            # Track loss and relevant metrics
            total_train_loss = seg_loss + d_loss
            total_seg_loss += seg_loss
            total_depth_loss += d_loss
            seg_len += len(segmentation)
            depth_len += len(depth)

            # Backpropagation
            total_train_loss.backward()
            optimizer.step()

            _, seg_pred = torch.max(segmentation_pred, 1)

            # Add metrics for current batch
            train_metrics.add(seg_pred, segmentation, depth_pred, depth)
        
            global_step += 1

        # Compute epoch-wide metrics for training
        train_epoch_metrics = train_metrics.compute()

        # Compute average losses for the epoch
        avg_seg_loss = total_seg_loss / seg_len
        avg_depth_loss = total_depth_loss / depth_len
        avg_train_loss = total_train_loss / len(train_data)

        # Store metrics for the epoch
        metrics["train"]["total_depth_loss"].append(avg_depth_loss.item())
        metrics["train"]["total_seg_loss"].append(avg_seg_loss.item())
        metrics["train"]["total_loss"].append(avg_train_loss.item())
        metrics["train"]["iou"].append(train_epoch_metrics["iou"])
        metrics["train"]["abs_depth_error"].append(train_epoch_metrics["abs_depth_error"])
        metrics["train"]["tp_depth_error"].append(train_epoch_metrics["tp_depth_error"])

        # Log metrics to tensorboard
        logger.add_scalar("train/total_loss", total_train_loss.item(), global_step)
        logger.add_scalar("train/iou", train_epoch_metrics["iou"], global_step)
        logger.add_scalar("train/abs_depth_error", train_epoch_metrics["abs_depth_error"], global_step)
        logger.add_scalar("train/tp_depth_error", train_epoch_metrics["tp_depth_error"], global_step)
        logger.add_scalar("train/seg_loss", avg_seg_loss.item(), global_step)
        logger.add_scalar("train/depth_loss", avg_depth_loss.item(), global_step)
        
        # Disable gradient compution and switch to evaluation mode
        model.eval()
        with torch.inference_mode():
            # Reset metrics
            val_metrics.reset()
            total_val_loss = 0
            total_seg_loss = 0
            seg_len = 0
            total_depth_loss = 0
            depth_len = 0

            for batch in val_data:
                # Put img depth and segmentation data on GPU
                img = batch["image"].to(device)
                depth = batch["depth"].to(device)
                segmentation = batch["track"].to(device)

                # Predict depth and segmentation
                segmentation_pred, depth_pred = model(img)
                
                # Compute losses
                seg_loss = segmentation_loss(segmentation_pred, segmentation)
                d_loss = depth_loss(depth_pred, depth)

                # Track losses and relevant metrics
                total_val_loss = seg_loss + d_loss
                total_seg_loss += seg_loss
                total_depth_loss += d_loss
                seg_len += len(segmentation)
                depth_len += len(depth)

                _, seg_pred = torch.max(segmentation_pred, 1)

                # Add metrics for current batch
                val_metrics.add(seg_pred, segmentation, depth_pred, depth)

            # Compute epoch-wide metrics for validation
            val_epoch_metrics = val_metrics.compute()

            # Compute average losses for the epoch
            avg_seg_loss = total_seg_loss / seg_len
            avg_depth_loss = total_depth_loss / depth_len
            avg_val_loss = total_val_loss / len(val_data)

            # Store metrics for the epoch
            metrics["val"]["total_depth_loss"].append(avg_depth_loss.item())
            metrics["val"]["total_seg_loss"].append(avg_seg_loss.item())
            metrics["val"]["total_loss"].append(avg_val_loss.item())
            metrics["val"]["iou"].append(val_epoch_metrics["iou"])
            metrics["val"]["abs_depth_error"].append(val_epoch_metrics["abs_depth_error"])
            metrics["val"]["tp_depth_error"].append(val_epoch_metrics["tp_depth_error"])

            # Log metrics to tensorboard
            logger.add_scalar("val/total_loss", avg_val_loss.item(), global_step)
            logger.add_scalar("val/iou", val_epoch_metrics["iou"], global_step)
            logger.add_scalar("val/abs_depth_error", val_epoch_metrics["abs_depth_error"], global_step)
            logger.add_scalar("val/tp_depth_error", val_epoch_metrics["tp_depth_error"], global_step)
            logger.add_scalar("val/seg_loss", avg_seg_loss.item(), global_step)
            logger.add_scalar("val/depth_loss", avg_depth_loss.item(), global_step)

        # Print on first, last and every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}: "
                f"Train IoU={train_epoch_metrics['iou']:.4f} "
                f"Val IoU={val_epoch_metrics['iou']:.4f}"
            )

    # Save and overwrite the model in the root directory
    save_model(model)

    # Save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    # Define the arguments for train_detection
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2024)

    # Pass all arguments to train_detection
    train_detection(**vars(parser.parse_args()))
