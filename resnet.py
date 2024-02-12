"""
Implementation of ResNet18 Model using PyTorch
"""

import torch
from torch import nn

class BasicBlock(nn.Module): #pylint: disable=too-few-public-methods
    """
    Basic ResNet block, composed of two conv layers, each followed by BatchNorm and ReLU activation.
    Optionally includes a downsampling layer.

    Attributes:
        expansion (int):    Multiplier for the number of output channels in the block.
                            For BasicBlock, it is always 1.
    """
    expansion=1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Initializes the BasicBlock.

        Parameters:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int): Stride size for the first convolutional layer.
            downsample (nn.Module, optional): Downsampling module for adjusting dimensions.
            padding (int): Padding for the convolutional layers.
        """
        super().__init__()

        # Check that downsampling is defined if dims don't match
        if downsample is None and (stride != 1 or inplanes != planes * self.expansion):
            raise ValueError("Downsample cannot be None for inplanes!=planes*expansion")

        self.conv1 = nn.Conv2d(inplanes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # ReLU used after each BatchNorm
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        #Optional downsampling to match dimensions
        self.downsample=downsample
        self.stride=stride

    def forward(self, x):
        """
        Forward pass of the BasicBlock.

        Parameters:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the block.
        """

        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        #Adding residual (identity) conection
        out += identity
        return self.relu(out)

class ResNet(nn.Module): #pylint: disable=too-few-public-methods, too-many-instance-attributes
    """
    Implementation of the ResNet architecture.

    Attributes:
        inplanes (int): Number of channels in the input layer.
    """
    def __init__(self, block, layers, num_classes=10):
        """
        Initializes the ResNet model.

        Parameters:
            block (nn.Module): Block type to be used (BasicBlock/Bottleneck (not implemented here)).
            layers (list): Number of blocks in each layer of the network.
            num_classes (int): Number of classes for the final output layer.
        """
        super().__init__()
        # Inital number of planes
        self.inplanes = 32

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)

        # Sequential layers
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool= nn.AdaptiveAvgPool2d((1,1))
#         self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(256*block.expansion, num_classes)

        # Apply He Initialization. "fan out" is recommended for ReLU,
        # preserves magnitudes in backward pass
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Creates a layer composed of multiple blocks.

        Parameters:
            block (nn.Module): The class representing the block.
            planes (int): Number of output channels for each block.
            blocks (int): Number of blocks in the layer.
            stride (int): Stride size for the first block in the layer.

        Returns:
            nn.Sequential: A sequential container of blocks forming a layer.
        """
        downsample=None
        # Check whether we need to adjust dimensions
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
        )

        layers = [block(self.inplanes, planes, stride, downsample)]
        # Update inplanes with block expansion
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNet model.

        Parameters:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor of the model.
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18():
    """
    Constructs a ResNet-18 model.

    Returns:
        ResNet: An instance of the ResNet model with 18 layers.
    """
    return ResNet(BasicBlock, [2,2,2,2])
