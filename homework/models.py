from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):

    class Block(torch.nn.Module):

        # AI: I was using GroupNorm before and it told me BatchNorm is better
        # AI: I was only using 1 Conv2d layer before and it told me to use 2
        # AI: It told me to add dropout and how much because I was having issues with my validation accuracy
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2
            self.c1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
            self.c2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.ReLU()
            self.skip = torch.nn.Conv2d(in_channels, out_channels, 1, stride, 0) if in_channels != out_channels else torch.nn.Identity()
            self.dropout = torch.nn.Dropout(0.4)
        

        # AI: Recommended to add skip aconnection before relu because I had it after in the return statement
        # AI: It told me to add dropout and how much because I was having issues with my validation accuracy
        def forward(self, x):
            res = self.skip(x)
            x = self.relu(self.bn1(self.c1(x)))
            x = self.relu(self.bn2(self.c2(x)))
            x = self.dropout(x)
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

        # AI: It recommended me to increase the channels and blocks
        channels_l0 = 24
        n_blocks = 5

        cnn_layers = [
            # AI: Recommended me to use kernel size of 7
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

        out = self.cnn(z)
        # AI: Recommened to use view because I was just directly returning out without doing so, which lead to errors
        out = out.view(out.size(0), -1)
        return out


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


        def forward(self, x):
            x = self.relu1(self.bn1(self.c1(x)))
            return x
    
    # AI: Showed me how to use ConvTranspose2d
    class UpBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.ct1 = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
            self.relu1 = torch.nn.ReLU()


        def forward(self, x):
            x = self.relu1(self.bn1(self.ct1(x)))
            return x


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
        
        # AI: Recommended me to increase the in channels
        # Downsampling path (encoder)
        self.down1 = self.DownBlock(in_channels, 32) # Output: (B, 32, 48, 64)
        self.down2 = self.DownBlock(32, 64) # Output: (B, 64, 24, 32)
        self.down3 = self.DownBlock(64, 128)  # Output: (B, 128, 12, 16)
        self.down4 = self.DownBlock(128, 256) # Output: (B, 256, 6, 8)
        self.down5 = self.DownBlock(256, 512) # Output: (B, 512, 3, 4)

        # AI: Recommended me to increase the out channels
        # Upsampling path (decoder)
        self.up1 = self.UpBlock(512, 256) # Output: (B, 256, 6, 8)
        self.up2 = self.UpBlock(256, 128) # Output: (B, 128, 12, 16)
        self.up3 = self.UpBlock(128, 64) # Output: (B, 64, 24, 32)
        self.up4 = self.UpBlock(64, 32) # Output: (B, 32, 48, 64)
        self.up5 = self.UpBlock(32, 32) # Output: (B, 32, 96, 128)

        # AI: Showed me how to get the segmentation head and depth head as I didn't know how to do it
        # Segmentation head
        self.segmentation_head = torch.nn.Conv2d(32, num_classes, kernel_size=1) # Output: (B, num_classes, 96, 128)

        # Depth head
        self.depth_head = torch.nn.Conv2d(32, 1, kernel_size=1) # Output: (B, 1, 96, 128)

        # AI: Showed me the correct input and output channels to use for skip connections
        # Skip connections
        self.skip1 = torch.nn.Conv2d(32, 32, kernel_size=1)
        self.skip2 = torch.nn.Conv2d(64, 64, kernel_size=1)
        self.skip3 = torch.nn.Conv2d(128, 128, kernel_size=1)
        self.skip4 = torch.nn.Conv2d(256, 256, kernel_size=1)


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
        
        # Downsample (encoder)
        down1 = self.down1(z) # Output: (B, 32, 48, 64)
        down2 = self.down2(down1) # Output: (B, 64, 24, 32)
        down3 = self.down3(down2) # Output: (B, 128, 12, 16)
        down4 = self.down4(down3) # Output: (B, 256, 6, 8)
        down5 = self.down5(down4) # Output: (B, 512, 3, 4)

        # AI: I had problems with where to put the right skip connections, it showed me where to put them
        # Upsample (decoder) with skip connections
        up1 = self.up1(down5) + self.skip4(down4) # Output: (B, 256, 6, 8)
        up2 = self.up2(up1) + self.skip3(down3) # Output: (B, 128, 12, 16)
        up3 = self.up3(up2) + self.skip2(down2) # Output: (B, 64, 24, 32)
        up4 = self.up4(up3) + self.skip1(down1) # Output: (B, 32, 48, 64)
        up5 = self.up5(up4) # Output: (B, 32, 96, 128)

        # Segmentation head
        segmentation_out = self.segmentation_head(up5) # Output: (B, num_classes, 96, 128)

        # AI: Showed me which up to use for depth head
        # Depth head
        depth_out = self.depth_head(up5) # Output: (B, 1, 96, 128)
        depth_out = depth_out.squeeze(1) # Output: (B, 96, 128)

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
