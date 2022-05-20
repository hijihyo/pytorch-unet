"""
PyTorch U-Net and modules for it
"""
from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor


def conv_batch_relu(
    in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
):
    """Conv2d - BatchNorm2d - ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class UNetEncoderBlock(nn.Module):
    """Constructs a block which an encoder of U-Net consists of"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ) -> None:
        super(UNetEncoderBlock, self).__init__()

        conv_kwargs = {"kernel_size": kernel_size,
                       "stride": stride, "padding": padding}
        self.layer1 = conv_batch_relu(in_channels, out_channels, **conv_kwargs)
        self.layer2 = conv_batch_relu(
            out_channels, out_channels, **conv_kwargs)
        self.downsample = nn.MaxPool2d(kernel_size=2)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """Args:
            input: torch.Tensor, (C, H, W) or (B, C, H, W)

        Returns:
            output: torch.Tensor, (C', H/2, W/2) or (B, C', H/2, W/2)
            conv_output: torch.Tensor, (C', H, W) or (B, C', H, W)
        """
        conv_output = self.layer1(input)
        conv_output = self.layer2(conv_output)
        output = self.downsample(conv_output)
        return output, conv_output


class UNetDecoderBlock(nn.Module):
    """Constructs a block which an decoder of U-Net consists of"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
    ) -> None:
        super(UNetDecoderBlock, self).__init__()

        conv_kwargs = {"kernel_size": kernel_size,
                       "stride": stride, "padding": padding}
        # TODO: mode of `nn.Upsample`?
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        # TODO: i've changed the structure
        self.layer1 = conv_batch_relu(2 * in_channels,
                                      in_channels, **conv_kwargs)
        self.layer2 = conv_batch_relu(
            in_channels, out_channels, **conv_kwargs)

    def forward(self, input: Tensor, conv_output: Tensor) -> Tensor:
        """Args:
            input: torch.Tensor, (C, H, W) or (B, C, H, W)
            conv_output: torch.Tensor, (C, H*2, W*2) or (B, C, H*2, W*2)

        Returns:
            output: torch.Tensor, (C', H*2, W*2) or (B, C', H*2, W*2)
        """
        upsampled = self.upsample(input)  # (C, H*2, W*2)
        combined = torch.cat((upsampled, conv_output),
                             dim=input.dim()-3)  # (C*2, H*2, W*2)
        output = self.layer1(combined)
        output = self.layer2(output)
        return output


class UNetEncoder(nn.Module):
    """Constructs an encoder of U-Net"""

    def __init__(
        self, in_channels: int, channels: int, kernel_size: int,
        stride: int, padding: int, dropout: float = 0.
    ) -> None:
        super(UNetEncoder, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = None

        conv_kwargs = {"kernel_size": kernel_size,
                       "stride": stride, "padding": padding}
        self.blocks = nn.ModuleList([
            UNetEncoderBlock(
                in_channels if i == 0 else channels[i - 1],
                channels[i],
                **conv_kwargs
            )
            for i in range(len(channels) - 1)
        ])
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.block_output = conv_batch_relu(
            channels[-2], channels[-1], **conv_kwargs)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """Args:
            input: torch.Tensor, (C, H, W) or (B, C, H, W)

        Returns:
            output: torch.Tensor, (C', H', W') or (B, C', H', W')
            conv_output_list: a list of torch.Tensor
        """
        output = input
        conv_output_list = []
        for i, _ in enumerate(self.blocks):
            output, conv_output = self.blocks[i](output)
            conv_output_list.append(conv_output)
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.block_output(output)
        return output, conv_output_list


class UNetDecoder(nn.Module):
    """Constructs an decoder of U-Net"""

    def __init__(
        self, channels: int, num_classes: int, kernel_size: int,
        stride: int, padding: int, dropout: float = 0.
    ) -> None:
        super(UNetDecoder, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = None

        conv_kwargs = {"kernel_size": kernel_size,
                       "stride": stride, "padding": padding}
        self.block_input = conv_batch_relu(
            channels[0], channels[1], **conv_kwargs)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            UNetDecoderBlock(
                channels[i],
                num_classes if i + 1 == len(channels) else channels[i + 1],
                **conv_kwargs
            )
            for i in range(1, len(channels))
        ])

    def forward(self, input: Tensor, conv_output_list: List[Tensor]) -> Tensor:
        """Args:
            input: torch.Tensor, (C, H, W) or (B, C, H, W)
            conv_output_list: a list of torch.Tensor

        Returns:
            output: torch.Tensor, (C', H', W') or (B, C', H', W')
        """
        output = self.block_input(input)
        if self.dropout is not None:
            output = self.dropout(output)
        for i, _ in enumerate(self.blocks):
            output = self.blocks[i](output, conv_output_list[i])
        return output


class UNet(nn.Module):
    """Constructs U-Net"""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.) -> None:
        super(UNet, self).__init__()
        self.kernel_size = 3
        self.stride = 1
        self.padding = (self.kernel_size - self.stride) // 2
        self.channels = [64, 128, 256, 512, 1024]

        self.encoder = UNetEncoder(
            in_channels, self.channels, self.kernel_size, self.stride, self.padding, dropout)
        self.decoder = UNetDecoder(
            self.channels[::-1], num_classes, self.kernel_size, self.stride, self.padding, dropout)

    def forward(self, input: Tensor) -> Tensor:
        """Args:
            input: torch.Tensor, (C, H, W) or (B, C, H, W)

        Returns:
            output: torch.Tensor, (C', H, W) or (B, C', H, W)
        """
        output, conv_output_list = self.encoder(input)
        output = self.decoder(output, conv_output_list[::-1])
        return output
