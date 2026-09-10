from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class Refinement(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_f_maps: int,
        dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(DilatedResidualLayer(2**index, num_f_maps, num_f_maps))
                for index in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        return self.conv_out(out)


class PredictionGeneration(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_f_maps: int,
        dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, kernel_size=1)
        self.conv_dilated_1 = nn.ModuleList(
            nn.Conv1d(
                num_f_maps,
                num_f_maps,
                kernel_size=3,
                padding=2 ** (num_layers - 1 - index),
                dilation=2 ** (num_layers - 1 - index),
            )
            for index in range(num_layers)
        )
        self.conv_dilated_2 = nn.ModuleList(
            nn.Conv1d(
                num_f_maps,
                num_f_maps,
                kernel_size=3,
                padding=2**index,
                dilation=2**index,
            )
            for index in range(num_layers)
        )
        self.conv_fusion = nn.ModuleList(
            nn.Conv1d(2 * num_f_maps, num_f_maps, kernel_size=1)
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_1x1_in(x)
        for index in range(self.num_layers):
            residual = features
            features = self.conv_fusion[index](
                torch.cat(
                    [
                        self.conv_dilated_1[index](features),
                        self.conv_dilated_2[index](features),
                    ],
                    dim=1,
                )
            )
            features = F.relu(features)
            features = self.dropout(features)
            features = features + residual
        return self.conv_out(features)


class MS_TCN2(nn.Module):
    def __init__(
        self,
        num_layers_PG: int,
        num_layers_R: int,
        num_R: int,
        num_f_maps: int,
        dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = dim
        self.PG = PredictionGeneration(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList(
            [
                copy.deepcopy(
                    Refinement(num_layers_R, num_f_maps, num_classes, num_classes)
                )
                for _ in range(num_R)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for refinement in self.Rs:
            out = refinement(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs
