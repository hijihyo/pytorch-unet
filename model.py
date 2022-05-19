"""
PyTorch U-Net model and modules for the model
"""
from typing import List
import torch
from torch import nn


class UNetEncoderBlock(nn.Module):
    """Constructs a encoder block of a U-Net model"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int):
        super(UNetEncoderBlock, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(pool_size)

    def forward(self, input: torch.Tensor):
        """Args:
            input: a `torch.Tensor`, (in_channels, width, height) or
            (batch_size, in_channels, width, height)

        Returns:
            output: a `torch.Tensor`, (out_channels, width', height') or
            (batch_size, out_channels, width', height')
            conv_output: a `torch.Tensor`, (out_channels, width',
            height') or (batch_size, out_channels, width'', height'')
        """
        conv_output = self.convs(input)
        output = self.maxpool(conv_output)
        return output, conv_output


class UNetDecoderBlock(nn.Module):
    """Constructs a decoder block of a U-Net model"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int):
        super(UNetDecoderBlock, self).__init__()

        self.upconv = nn.ConvTranspose2d(
            in_channels, in_channels // 2, pool_size, stride=2)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, input: torch.Tensor, ec_output: torch.Tensor):
        """Args:
            input: a `torch.Tensor`, (in_channels, width, height) or
            (batch_size, in_channels, width, height)
            ec_output: an output `torch.Tensor` from the same-level of
            the encoder, (in_channels // 2, width', height') or
            (batch_size, in_channels // 2, width', height')

        Returns:
            output: a `torch.Tensor`, (out_channels, width', height') or
            (batch_size, out_channels, width', height')
        """
        is_batched = input.dim() == 4
        if not is_batched:
            input = input.unsqueeze(0)
            ec_output = ec_output.unsqueeze(0)
        output = self.upconv(input)
        width_diff = ec_output.size(2) - output.size(2)
        height_diff = ec_output.size(3) - output.size(3)
        eo_cropped = ec_output[:, :,
                               width_diff//2:-width_diff//2,
                               height_diff//2:-height_diff//2
                               ]
        output = torch.concat((output, eo_cropped), dim=1)
        output = self.convs(output)
        if not is_batched:
            output = output.squeeze(0)
        return output


class UNetEncoder(nn.Module):
    """Constructs a encoder of a U-Net model"""

    def __init__(
        self, in_channels: int, blocks: List[int], out_channels: int,
        kernel_size: int, pool_size: int, dropout: float = 0.
    ):
        super(UNetEncoder, self).__init__()

        self.blocks = nn.ModuleList(
            [UNetEncoderBlock(in_channels if i == 0 else blocks[i - 1],
                              blocks[i], kernel_size, pool_size) for i, _ in enumerate(blocks)]
        )
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.output_block = nn.Sequential(
            nn.Conv2d(blocks[-1], out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, input: torch.Tensor):
        """Args:
            input: a `torch.Tensor`, (in_channels, width, height) or
            (batch_size, in_channels, width, height)

        Returns:
            output: a `torch.Tensor`, (out_channels, width', height') or
            (batch_size, out_channels, width, height)
            block_output: a list of `torch.Tensor`s (length: num_blocks)
        """
        output = input
        block_output = []
        for i, _ in enumerate(self.blocks):
            output, conv_output = self.blocks[i](output)
            block_output.append(conv_output)
        if self.dropout is not None:
            output = self.dropout(output)  # 원래 위치는 아님
        output = self.output_block(output)
        return output, block_output


class UNetDecoder(nn.Module):
    """Constructs a decoder of a U-Net model"""

    def __init__(
        self, in_channels: int, blocks: List[int], out_channels: int,
        kernel_size: int, pool_size: int, dropout: float = 0.
    ):
        super(UNetDecoder, self).__init__()

        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [UNetDecoderBlock(in_channels if i == 0 else blocks[i - 1],
                              blocks[i], kernel_size, pool_size) for i, _ in enumerate(blocks)]
        )
        self.output_block = nn.Conv2d(blocks[-1], out_channels, kernel_size=1)

    def forward(self, input: torch.Tensor, ec_output: List[torch.Tensor]):
        """Args:
            input: a `torch.Tensor`, (in_channels, width, height) or
            (batch_size, in_channels, width, height)
            ec_output: a list of `torch.Tensor`s (length: num_blocks)
        """
        output = self.input_block(input)
        if self.dropout is not None:
            output = self.dropout(output)
        for i, _ in enumerate(self.blocks):
            output = self.blocks[i](output, ec_output[- i - 1])
        output = self.output_block(output)
        return output


class UNet(nn.Module):
    """Constructs a U-Net model"""

    def __init__(self, channels: List[int], num_classes: int, dropout: float = 0.):
        super(UNet, self).__init__()
        self.kernel_size = 3
        self.pool_size = 2

        self.encoder = UNetEncoder(
            channels[0], channels[1:-1], channels[-1], self.kernel_size, self.pool_size, dropout)
        self.decoder = UNetDecoder(
            channels[-1], channels[-2:0:-1], num_classes, self.kernel_size, self.pool_size, dropout)

    def forward(self, input: torch.Tensor):
        """Args:
            input: a `torch.Tensor`, (in_channels, width, height) or
            (batch_size, in_channels, width, height)

        Returns:
            output: a `torch.Tensor`, (out_channels, width', height') or
            (batch_size, out_channels, width, height)
        """
        output, block_output = self.encoder(input)
        output = self.decoder(output, block_output)
        return output
