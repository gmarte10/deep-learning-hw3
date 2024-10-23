from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2
            self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.bn1 = torch.nn.BatchNorm2d(out_channels) # AI, I was using GroupNorm before
            self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.ReLU()
            self.skip = torch.nn.Conv2d(in_channels, out_channels, 1, stride, 0) if in_channels != out_channels else torch.nn.Identity()
            self.dropout = torch.nn.Dropout(0.3) # AI, Didn't have dropout before
        
        def forward(self, x):
            res = self.skip(x) # AI, had it in return statement before
            x = self.relu(self.bn1(self.c1(x)))
            x = self.relu(self.bn2(self.c2(x)))
            x = self.dropout(x) # AI, Didn't have dropout before
            return self.relu(x + res)
        
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # TODO: implement - saving
        channels_l0 = 96
        n_blocks = 3

        cnn_layers = [
            torch.nn.Conv2d(in_channels, channels_l0, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(channels_l0),
            torch.nn.ReLU(),
        ]
        c1 = channels_l0
        for _ in range(n_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride=2))
            c1 = c2
        cnn_layers.append(torch.nn.Conv2d(c1, num_classes, kernel_size=1))
        self.cnn = torch.nn.Sequential(*cnn_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        out = self.cnn(z)
        out = out.view(out.size(0), -1)
        return out
        # logits = torch.randn(x.size(0), 6)

        # return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):

    class DownBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
            self.relu1 = torch.nn.ReLU()
            self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            self.relu2 = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(0.1)

        def forward(self, x):
            x = self.relu1(self.bn1(self.c1(x)))
            x = self.relu2(self.bn2(self.c2(x)))
            x = self.dropout(x)
            return x
        
    class UpBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.ct1 = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
            self.relu1 = torch.nn.ReLU()
            self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            self.relu2 = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(0.1)

        def forward(self, x):
            x = self.relu1(self.bn1(self.ct1(x)))
            x = self.relu2(self.bn2(self.c2(x)))
            x = self.dropout(x)
            return x

        # def forward(self, x, skip_connection=None):
        #     x = self.relu(self.bn1(self.ct1(x)))
        #     if skip_connection is not None:
        #         x = x + skip_connection
        #     return x

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))
        
        # Downsampling path (encoder)
        self.down1 = self.DownBlock(in_channels, 16) # Output: (B, 16, 48, 64)
        self.down2 = self.DownBlock(16, 32) # Output: (B, 32, 24, 32)
        self.down3 = self.DownBlock(32, 64) # Output: (B, 64, 12, 16)
        self.down4 = self.DownBlock(64, 128) # Output: (B, 128, 6, 8)
        self.down5 = self.DownBlock(128, 256) # Output: (B, 256, 3, 4)
        self.down6 = self.DownBlock(256, 512) # Output: (B, 512, 2, 2)

        # Upsampling path (decoder)
        self.up1 = self.UpBlock(512, 256) # Output: (B, 256, 3, 4)
        self.up2 = self.UpBlock(256, 128) # Output: (B, 128, 6, 8)
        self.up3 = self.UpBlock(128, 64) # Output: (B, 64, 12, 16)
        self.up4 = self.UpBlock(64, 32) # Output: (B, 32, 24, 32)
        self.up5 = self.UpBlock(32, 16) # Output: (B, 16, 48, 64)
        self.up6 = self.UpBlock(16, 16) # Output: (B, 16, 96, 128)

        # Segmentation head
        self.segmentation_head = torch.nn.Conv2d(16, num_classes, kernel_size=1) # Output: (B, num_classes, 96, 128)

        # Depth head
        self.depth_head = torch.nn.Conv2d(16, 1, kernel_size=1) # Output: (B, 1, 96, 128)

        # Skip connections
        self.skip1 = torch.nn.Conv2d(16, 16, kernel_size=1)
        self.skip2 = torch.nn.Conv2d(32, 32, kernel_size=1)
        self.skip3 = torch.nn.Conv2d(64, 64, kernel_size=1)
        self.skip4 = torch.nn.Conv2d(128, 128, kernel_size=1)
        self.skip5 = torch.nn.Conv2d(256, 256, kernel_size=1)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None] # (B, 3, 96, 128)
        # print("Shape after z:", z.shape)

        # Downsample (encoder)
        down1 = self.down1(z) # (B, 16, 48, 64)
        # print("Shape after down1:", down1.shape)
        down2 = self.down2(down1) # (B, 32, 24, 32)
        # print("Shape after down2:", down2.shape)
        down3 = self.down3(down2) # (B, 64, 12, 16)
        # print("Shape after down3:", down3.shape)
        down4 = self.down4(down3) # (B, 128, 6, 8)
        # print("Shape after down4:", down4.shape)
        down5 = self.down5(down4) # (B, 256, 3, 4)
        # print("Shape after down5:", down5.shape)
        down6 = self.down6(down5) # (B, 512, 2, 2)

        # Upsample (decoder)
        up1 = self.up1(down6) + self.skip5(down5) # (B, 256, 3, 4)
        # print("Shape after down6:", down6.shape)
        up2 = self.up2(up1) + self.skip4(down4) # (B, 128, 6, 8)
        # print("Shape after up1:", up1.shape)
        up3 = self.up3(up2) + self.skip3(down3) # (B, 64, 12, 16)
        # print("Shape after up2:", up2.shape)
        up4 = self.up4(up3) + self.skip2(down2) # (B, 32, 24, 32)
        # print("Shape after up3:", up3.shape)
        up5 = self.up5(up4) + self.skip1(down1) # (B, 16, 48, 64)
        # print("Shape after up4:", up4.shape)
        up6 = self.up6(up5) # (B, 16, 96, 128)

        # up1 = self.up1(down3) # (B, 32, 24, 32)
        # up2 = self.up2(up1) # (B, 16, 48, 64)
        # up3 = self.up3(up2) # (B, 16, 96, 128)

        # Segmentation head
        segmentation_out = self.segmentation_head(up6) # (B, num_classes, 96, 128)
        # print("Shape after segmentation_out:", segmentation_out.shape)

        # Depth head
        depth_out = self.depth_head(up6) # (B, 1, 96, 128)
        # print("Shape after depth_out:", depth_out.shape)

        return segmentation_out, depth_out


    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
