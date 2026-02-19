# This file is part of Dynamic Affine Feature Map Transform (DAFT).
#
# DAFT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DAFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DAFT. If not, see <https://www.gnu.org/licenses/>.
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..models.base import BaseModel
from .vol_blocks import ConvBnReLU, DAFTBlock, FilmBlock, ResBlock
from .vol_blocks import BasicBlock, Bottleneck
import numpy as np
import random 
from functools import partial

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


def conv3d(in_channels, out_channels, kernel_size=3, stride=1):
    if kernel_size != 1:
        padding = 1
    else:
        padding = 0
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)


class HeterogeneousResNet(BaseModel):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, n_basefilters=4) -> None:
        super().__init__()

        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return {"logits": out}

class ResNet2(BaseModel):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, n_basefilters=4) -> None:
        super().__init__()

        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(4 * n_basefilters, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return {"logits": out}

class ResNet3(BaseModel):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, n_basefilters=4) -> None:
        super().__init__()

        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(2 * n_basefilters, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return {"logits": out}


class ResNet4(BaseModel):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, n_basefilters=4) -> None:
        super().__init__()

        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.block5 = ResBlock(8 * n_basefilters, 16 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 2
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(16 * n_basefilters, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return {"logits": out}


class FCONLY(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
        bottleneck_dim: int = 7,
    ):

        super().__init__()
        # layers = [
        #     ("fc1", nn.Linear(ndim_non_img, bottleneck_dim)),
        #     # ("dropout", nn.Dropout(p=0.5, inplace=True)),
        #     ("relu", nn.ReLU()),
        #     ("fc2", nn.Linear(bottleneck_dim, n_outputs)),
        # ]
        # self.fc = nn.Sequential(OrderedDict(layers))

        self.fc1 = nn.Linear(ndim_non_img, bottleneck_dim)  # 4
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(bottleneck_dim, n_outputs)


    @property
    def input_names(self) -> Sequence[str]:
        # return ("image", "tabular")
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        # out = self.fc(tabular)

        out = self.fc1(tabular)
        out = self.act(out)
        out = self.fc2(out)

        return {"logits": out}



class FCONLY2(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
        bottleneck_dim: int = 7,
    ):

        super().__init__()
        # layers = [
        #     ("fc1", nn.Linear(ndim_non_img, bottleneck_dim)),
        #     # ("dropout", nn.Dropout(p=0.5, inplace=True)),
        #     ("relu", nn.ReLU()),
        #     ("fc2", nn.Linear(bottleneck_dim, n_outputs)),
        # ]
        # self.fc = nn.Sequential(OrderedDict(layers))

        self.fc1 = nn.Linear(ndim_non_img, bottleneck_dim)  # 4
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(bottleneck_dim, n_outputs)


    @property
    def input_names(self) -> Sequence[str]:
        # return ("image", "tabular")
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        # out = self.fc(tabular)

        out = self.fc1(tabular)
        out = self.dropout(out)
        out = self.act(out)
        out = self.fc2(out)

        return {"logits": out}


class ConcatHNN1FC(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
    ) -> None:

        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters + ndim_non_img, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        print (out.shape)
        out = self.block4(out)
        print (out.shape)
        out = self.global_pool(out)
        print (out.shape)
        out = out.view(out.size(0), -1)
        print (out.shape)
        out = nn.functional.normalize(out, dim = 1)
        out = torch.cat((out, tabular), dim=1)
        out = self.fc(out)

        return {"logits": out}




class ConcatHNN2FC(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
        bottleneck_dim: int = 7,
    ):

        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        # layers = [
        #     ("fc1", nn.Linear(8 * n_basefilters + ndim_non_img, bottleneck_dim)),
        #     # ("dropout", nn.Dropout(p=0.5, inplace=True)),
        #     ("relu", nn.ReLU()),
        #     ("fc2", nn.Linear(bottleneck_dim, n_outputs)),
        # ]

        # layers = [
        #     ("fc1", nn.Linear(8 * n_basefilters + ndim_non_img, 8*(8 * n_basefilters + ndim_non_img))),
        #     # ("dropout", nn.Dropout(p=0.5, inplace=True)),
        #     ("relu", nn.ReLU()),
        #     ("fc2", nn.Linear(8*(8 * n_basefilters + ndim_non_img), n_outputs)),
        # ]

        layers = [
            ("fc1", nn.Linear(8 * n_basefilters + ndim_non_img, 4*(8 * n_basefilters + ndim_non_img))),
            # ("dropout", nn.Dropout(p=0.5, inplace=True)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(4*(8 * n_basefilters + ndim_non_img), n_outputs)),
        ]

        self.fc = nn.Sequential(OrderedDict(layers))

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        # print (out.shape)
        out = self.block4(out)
        # print (out.shape)
        out = self.global_pool(out)
        # print (out.shape)
        out = out.view(out.size(0), -1)
        # print (out.shape)
        out = nn.functional.normalize(out, dim = 1)
        out = torch.cat((out, tabular), dim=1)

        out = self.fc(out)

        return {"logits": out}


class ConcatHNN3FC(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
        bottleneck_dim: int = 7,
    ):

        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)


        layers = [
            ("fc1", nn.Linear(8 * n_basefilters + ndim_non_img, 8*(8 * n_basefilters + ndim_non_img))),
            # ("dropout", nn.Dropout(p=0.5, inplace=True)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(8*(8 * n_basefilters + ndim_non_img), n_outputs)),
        ]



        self.fc = nn.Sequential(OrderedDict(layers))

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = nn.functional.normalize(out, dim = 1)
        out = torch.cat((out, tabular), dim=1)

        out = self.fc(out)

        return {"logits": out}



class ConcatHNN4FC(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
        bottleneck_dim: int = 7,
    ):

        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.pcablock = nn.Linear(8 * n_basefilters, n_basefilters)


        layers = [
            ("fc1", nn.Linear( n_basefilters + ndim_non_img, 4*(n_basefilters + ndim_non_img))),
            # ("dropout", nn.Dropout(p=0.5, inplace=True)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(4*( n_basefilters + ndim_non_img), n_outputs)),
        ]



        self.fc = nn.Sequential(OrderedDict(layers))

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.pcablock(out)
        out = nn.functional.normalize(out, dim = 1)
        out = torch.cat((out, tabular), dim=1)

        out = self.fc(out)

        return {"logits": out}



class ConcatHNN5FC(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
        bottleneck_dim: int = 7,
    ):

        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)


        layers = [
            ("fc1", nn.Linear(8 * n_basefilters + ndim_non_img, 16*(8 * n_basefilters + ndim_non_img))),
            # ("dropout", nn.Dropout(p=0.5, inplace=True)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(16*(8 * n_basefilters + ndim_non_img), n_outputs)),
        ]



        self.fc = nn.Sequential(OrderedDict(layers))

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = nn.functional.normalize(out, dim = 1)
        out = torch.cat((out, tabular), dim=1)

        out = self.fc(out)

        return {"logits": out}



class ConcatHNN6FC(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
        bottleneck_dim: int = 7,
    ):

        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.global_pool = nn.AdaptiveAvgPool3d(1)


        layers = [
            ("fc1", nn.Linear(4 * n_basefilters + ndim_non_img, 16*(4 * n_basefilters + ndim_non_img))),
            # ("dropout", nn.Dropout(p=0.5, inplace=True)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(16*(4 * n_basefilters + ndim_non_img), n_outputs)),
        ]

        self.fc = nn.Sequential(OrderedDict(layers))



    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = nn.functional.normalize(out, dim = 1)
        out = torch.cat((out, tabular), dim=1)

        out = self.fc(out)

        return {"logits": out}



class ConcatSCORE2FC(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
        bottleneck_dim: int = 7,
    ):

        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.linear = nn.Linear(8 * n_basefilters, 1)

        layers = [
            ("fc1", nn.Linear(1 + ndim_non_img, 8*(1 + ndim_non_img))),
            # ("dropout", nn.Dropout(p=0.5, inplace=True)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(8*(1 + ndim_non_img), n_outputs)),
        ]

        self.fc = nn.Sequential(OrderedDict(layers))

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = torch.cat((out, tabular), dim=1)
        out = self.fc(out)

        return {"logits": out}


class CNN1FC(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
    ) -> None:

        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters + ndim_non_img, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, tabular), dim=1)
        out = self.fc(out)

        return {"logits": out}


class CNN2FC(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
        bottleneck_dim: int = 7,
    ):

        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, 8*n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32

        self.conv2 = conv3d(8*n_basefilters, 4 *n_basefilters, stride=1)
        self.bn2 = nn.BatchNorm3d(4 *n_basefilters, momentum=bn_momentum)
        self.conv3 = conv3d(4 *n_basefilters, 2 *n_basefilters, stride=1)
        self.bn3 = nn.BatchNorm3d(2 *n_basefilters, momentum=bn_momentum)
        self.conv4 = conv3d(2 *n_basefilters, n_basefilters, stride=1)
        self.bn4 = nn.BatchNorm3d(n_basefilters, momentum=bn_momentum)        
        self.relu = nn.ReLU(inplace=True)

        # self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        # self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        # self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        # self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)


        layers = [
            ("fc1", nn.Linear(n_basefilters + ndim_non_img, 4*( n_basefilters + ndim_non_img))),
            # ("dropout", nn.Dropout(p=0.5, inplace=True)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(4*(n_basefilters + ndim_non_img), n_outputs)),
        ]


        self.fc = nn.Sequential(OrderedDict(layers))

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)               
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, tabular), dim=1)
        out = self.fc(out)

        return {"logits": out}


class ConcatHNNMCM(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
        bottleneck_dim: int = 7,
    ) -> None:

        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Linear(ndim_non_img, 30*bottleneck_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8 * n_basefilters + 30*bottleneck_dim, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        tab_transformed = self.mlp(tabular)
        tab_transformed = self.relu(tab_transformed)
        out = torch.cat((out, tab_transformed), dim=1)
        out = self.fc(out)

        return {"logits": out}


class InteractiveHNN(BaseModel):
    """
    adapted version of Duanmu et al. (MICCAI, 2020)
    https://link.springer.com/chapter/10.1007%2F978-3-030-59713-9_24
    """

    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
    ) -> None:

        super().__init__()

        # ResNet
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

        layers = [
            ("aux_base", nn.Linear(ndim_non_img, 8, bias=False)),
            ("aux_relu", nn.ReLU()),
            # ("aux_dropout", nn.Dropout(p=0.2, inplace=True)),
            ("aux_1", nn.Linear(8, n_basefilters, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

        self.aux_2 = nn.Linear(n_basefilters, n_basefilters, bias=False)
        self.aux_3 = nn.Linear(n_basefilters, 2 * n_basefilters, bias=False)
        self.aux_4 = nn.Linear(2 * n_basefilters, 4 * n_basefilters, bias=False)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)

        attention = self.aux(tabular)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block1(out)

        attention = self.aux_2(attention)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block2(out)

        attention = self.aux_3(attention)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block3(out)

        attention = self.aux_4(attention)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block4(out)

        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return {"logits": out}


# class FilmHNN(BaseModel):
#     """
#     adapted version of Perez et al. (AAAI, 2018)
#     https://arxiv.org/abs/1709.07871
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         n_outputs: int,
#         bn_momentum: float = 0.1,
#         n_basefilters: int = 4,
#         filmblock_args: Optional[Dict[Any, Any]] = None,
#     ) -> None:
#         super().__init__()

#         if filmblock_args is None:
#             filmblock_args = {}

#         self.split_size = 4 * n_basefilters
#         self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
#         self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
#         self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
#         self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
#         self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
#         self.blockX = FilmBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
#         self.global_pool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Linear(8 * n_basefilters, n_outputs)

#     @property
#     def input_names(self) -> Sequence[str]:
#         return ("image", "tabular")

#     @property
#     def output_names(self) -> Sequence[str]:
#         return ("logits",)

#     def forward(self, image, tabular):
#         out = self.conv1(image)
#         out = self.pool1(out)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)
#         out = self.blockX(out, tabular)
#         out = self.global_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)

#         return {"logits": out}


class FilmHNN(BaseModel):
    """
    adapted version of Perez et al. (AAAI, 2018)
    https://arxiv.org/abs/1709.07871
    """

    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        filmblock_args: Optional[Dict[Any, Any]] = None,
    ) -> None:
        super().__init__()

        if filmblock_args is None:
            filmblock_args = {}

        self.split_size = 4 * n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = FilmBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, tabular)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return {"logits": out}


class DAFT(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        filmblock_args: Optional[Dict[Any, Any]] = None,
    ) -> None:
        super().__init__()

        if filmblock_args is None:
            filmblock_args = {}

        self.split_size = 4 * n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = DAFTBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, tabular)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return {"logits": out}

class DAFT2(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        filmblock_args: Optional[Dict[Any, Any]] = None,
    ) -> None:
        super().__init__()

        if filmblock_args is None:
            filmblock_args = {}

        self.split_size = 4 * n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        # self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = DAFTBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        # self.blockX = DAFTBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(4 * n_basefilters, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        # out = self.block3(out)
        out = self.blockX(out, tabular)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return {"logits": out}


class DAFTFC(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        filmblock_args: Optional[Dict[Any, Any]] = None,
    ) -> None:
        super().__init__()

        if filmblock_args is None:
            filmblock_args = {}

        self.split_size = 4 * n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = DAFTBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        # self.fc = nn.Linear(8 * n_basefilters, n_outputs)

        # layers = [
        #     ("fc1", nn.Linear(8 * n_basefilters + ndim_non_img, 4*(8 * n_basefilters + ndim_non_img))),
        #     # ("dropout", nn.Dropout(p=0.5, inplace=True)),
        #     ("relu", nn.ReLU()),
        #     ("fc2", nn.Linear(4*(8 * n_basefilters + ndim_non_img), n_outputs)),
        # ]

        layers = [
            ("fc1", nn.Linear(8 * n_basefilters, 8*(8 * n_basefilters))),
            # ("dropout", nn.Dropout(p=0.5, inplace=True)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(8*(8 * n_basefilters), n_outputs)),
        ]

        self.fc = nn.Sequential(OrderedDict(layers))

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, tabular)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return {"logits": out}



class ResNetTransfer(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 ndim_non_img: int,
                 n_outputs: int,
                 n_basefilters: int,
                 num_seg_classes = 1,
                 shortcut_type='B',
                 no_cuda = False):
        
        n_basefilters = n_basefilters[0]
        # ndim_non_img = ndim_non_img[0]
        n_outputs = n_outputs[0]
        
        print ("n_outputs", n_outputs)
        print ("n_basefilters", n_basefilters)
        print ("ndim_non_img", ndim_non_img)
        
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNetTransfer, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)


        self.global_pool = nn.AdaptiveAvgPool3d(1)

        

        # #  Visual feature length
        self.pca = nn.Linear(2048, 8 * n_basefilters)

        # pred_layers = [
        #     ("fc1", nn.Linear(8 * n_basefilters + ndim_non_img, 16*(8 * n_basefilters + ndim_non_img))),
        #     # ("dropout", nn.Dropout(p=0.5, inplace=True)),
        #     ("relu", nn.ReLU()),
        #     ("fc2", nn.Linear(16*(8 * n_basefilters + ndim_non_img), n_outputs)),
        # ]


        # self.fc = nn.Sequential(OrderedDict(pred_layers))




        # self.fc = nn.Linear(8 * n_basefilters + ndim_non_img, n_outputs)
        # self.fc = nn.Linear(2048 + ndim_non_img, n_outputs)

        # self.fc = nn.Linear(2048, n_outputs)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)



        # self.fc = nn.Linear(2048, 1024)



        # self.conv_seg = nn.Sequential(
        #                                 nn.ConvTranspose3d(
        #                                 512 * block.expansion,
        #                                 32,
        #                                 2,
        #                                 stride=2
        #                                 ),
        #                                 nn.BatchNorm3d(32),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv3d(
        #                                 32,
        #                                 32,
        #                                 kernel_size=3,
        #                                 stride=(1, 1, 1),
        #                                 padding=(1, 1, 1),
        #                                 bias=False), 
        #                                 nn.BatchNorm3d(32),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.Conv3d(
        #                                 32,
        #                                 num_seg_classes,
        #                                 kernel_size=1,
        #                                 stride=(1, 1, 1),
        #                                 bias=False) 
        #                                 )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)


    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)


    def forward(self, image, tabular):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.normalize(x, dim = 1)
        x = self.pca(x)
        # x = torch.cat((x, tabular), dim=1)

        x = self.fc(x)

        return {"logits": x}




