"""
original:
https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
"""
import torch.nn as nn
import math
from typing import Union, Tuple, List
from src.models_custom import AttentionPoolerV1, ChannelDropout1d


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=(k_size,), padding="same", bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)  # [N, C, 1]

        # Two different branches of ECA module
        # [N, C, 1] -transpose(-1, -2)-> [N, 1, C] -conv-> [N, 1, C] -transpose(-1, -2)-> [N, C, 1]
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, k_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=(k_size,),
            stride=(1,),
            padding="same"
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=(k_size,),
            stride=(1,),
            padding="same"
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.eca = eca_layer(planes, k_size)
        self.downsample = downsample

    def forward(self, x):
        """
        x: [N, D, T]
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class EcaResNet1D(nn.Module):
    """
    * убраны все преобразования до 4 слоёв
    * same padding
    """
    def __init__(
            self,
            block,
            layers,
            input_dim: int = 768,
            num_classes: int = 256,
            k_size: Union[Tuple[int], List[int]] = (31, 17, 11, 5),
            add_channel_dropout: bool = False,  # для обратной совместимости
            p_channel_dropout: float = 0.1
    ):
        self.inplanes = input_dim
        super().__init__()

        # channel dropout
        self.channel_dropout = None
        if add_channel_dropout:
            self.channel_dropout = ChannelDropout1d(p_channel_dropout)

        self.layer1 = self._make_layer(block, 128, layers[0], int(k_size[0]))
        self.layer2 = self._make_layer(block, 256, layers[1], int(k_size[1]))
        self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]))
        self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]))
        self.pooler = AttentionPoolerV1(512)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, k_size):
        downsample = None
        if self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=(1,),
                    stride=(1,),
                    padding="same",
                    bias=False
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, downsample, k_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size))

        return nn.Sequential(*layers)

    def forward(self, x, mask, **kwargs):
        """
        x - [N, T, D]
        """
        if self.channel_dropout is not None:
            x = self.channel_dropout(x)
        x = x.transpose(-1, -2)  # [N, D, T]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.transpose(-1, -2)  # [N, T, D]
        x = self.pooler(x, mask)
        x = self.fc(x)
        return x


def parse_kernel_size(x) -> List[int]:
    if isinstance(x, str):
        if "," in x:
            sep = ","
        elif "-" in x:
            sep = "-"
        else:
            raise
        return [int(k) for k in x.split(sep)]
    else:
        return [int(k) for k in x]


def eca_resnet18(
        input_dim: int = 768,
        k_size: Union[Tuple, str] = (31, 17, 11, 5),
        num_classes: int = 256,
        add_channel_dropout: bool = False,
        p_channel_dropout: float = 0.1
):
    """Constructs a ResNet-18 model.
    """
    model = EcaResNet1D(
        ECABasicBlock,
        [2, 2, 2, 2],
        input_dim=input_dim,
        num_classes=num_classes,
        k_size=parse_kernel_size(k_size),
        add_channel_dropout=add_channel_dropout,
        p_channel_dropout=p_channel_dropout
    )
    return model


def eca_resnet34(
        input_dim: int = 768,
        k_size: Union[Tuple, str] = (31, 17, 11, 5),
        num_classes: int = 256,
        add_channel_dropout: bool = False,
        p_channel_dropout: float = 0.1
):
    """Constructs a ResNet-34 model.
    """
    model = EcaResNet1D(
        ECABasicBlock,
        [3, 4, 6, 3],
        input_dim=input_dim,
        num_classes=num_classes,
        k_size=parse_kernel_size(k_size),
        add_channel_dropout=add_channel_dropout,
        p_channel_dropout=p_channel_dropout
    )
    return model
