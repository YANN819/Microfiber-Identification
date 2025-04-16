import math
import einops
import torch
import pandas as pd
from torch import nn, optim
import torch.nn.functional as F



class LayerNorm(nn.Module):
    r"""from ConvNext
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.parameter.Parameter(torch.ones(normalized_shape))
        self.bias = nn.parameter.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class CNN1D(nn.Module):
    def __init__(self, in_channel: int, out_class: int, dropout=0.1) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            #conv 1
            nn.Conv1d(in_channel, 4, kernel_size=11, stride=7),
            nn.BatchNorm1d(4),
            nn.GELU(),
            #conv 2
            nn.Conv1d(4, 16, kernel_size=7, stride=5),
            nn.BatchNorm1d(16),
            #conv 3
            nn.Conv1d(16, 64, kernel_size=5, stride=3),
            nn.BatchNorm1d(64),
            nn.Dropout(p=dropout),
            #conv 4
            nn.Conv1d(64, 256, kernel_size=3, stride=3),
            nn.BatchNorm1d(256),
            #conv 5
            nn.Conv1d(256, 1024, kernel_size=3, stride=3),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(p=dropout)     
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 2, 2048),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(2048, out_class),
        )

    def forward(self, x: torch.Tensor):
        x = self.convs(x)
        print("After convolution:", x.shape)
        x = einops.rearrange(x, "b d l -> b (d l)")
        print("After rearrange:", x.shape)
        x = self.classifier(x)
        print("After classifier:", x.shape)
        return x
        

class CNN1D(nn.Module):
    def __init__(self, in_channel: int, out_class: int, dropout=0.1) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            #conv 1
            nn.Conv1d(in_channel, 4, kernel_size=11, stride=5), # input (1, 2656) output (4, 529)
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),  # output (4, 264)
            #conv 2
            nn.Conv1d(4, 16, kernel_size=7, stride=3),  # output (16, 86)
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=3, stride=2),  # output (16, 42)
            #conv 3
            nn.Conv1d(16, 64, kernel_size=3, padding=1, stride=1),  # output (64, 42)
            nn.BatchNorm1d(64),
            #conv 4
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1),  # output (64, 42)
            nn.BatchNorm1d(64),
            #conv 5
            nn.Conv1d(64, 512, kernel_size=3, padding=1, stride=1),  # output (512, 42)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512 * 20, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, out_class),
        )

    def forward(self, x: torch.Tensor):
        x = self.convs(x)
        print("After convolution:", x.shape)
        x = einops.rearrange(x, "b d l -> b (d l)")
        print("After rearrange:", x.shape)
        x = self.classifier(x)
        print("After classifier:", x.shape)
        return x


class CNN2D(nn.Module):
    def __init__(self, in_channel: int, out_class: int, dropout=0.1, norm_eps: float = 1e-6) -> None:
        super().__init__()
        act_layer = lambda name="relu": nn.GELU() if name=="gelu" else nn.ReLU(inplace=True)
        norm_layer = nn.BatchNorm2d
        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=11, stride=3, padding=0),
            norm_layer(64),
            act_layer(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=7, stride=3, padding=0),
            norm_layer(192),
            act_layer(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=5, stride=3, padding=1),
            norm_layer(384),
            act_layer(),
            nn.Conv2d(384, 768, kernel_size=3, stride=1, padding=1),
            norm_layer(768),
            act_layer(),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(3 * 3 * 768, 2048),
            act_layer(),
            nn.Dropout(p=dropout),
            nn.Linear(2048, out_class),
        )

    def forward(self, x: torch.Tensor):
        print("Input shape before convolutions:", x.shape)
        x = self.convs(x)
        print("Output shape after convolutions:", x.shape)
        x = einops.rearrange(x, "b c h w -> b (c h w)")
        print("Output shape after rearrange:", x.shape)
        x = self.classifier(x)
        return x
