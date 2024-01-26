"""
Test suite for ResNet18 implementation for CIFAR10
"""
import pytest
import numpy as np
import torch
from torch import nn
from resnet import ResNet, BasicBlock, resnet18

def test_basic_block_initialization():
    """
    Test initialization of BasicBlock components.
    """
    block = BasicBlock(3, 64)
    assert isinstance(block.conv1, nn.Conv2d)
    assert isinstance(block.bn1, nn.BatchNorm2d)
    assert isinstance(block.relu, nn.ReLU)
    assert isinstance(block.conv2, nn.Conv2d)
    assert isinstance(block.bn2, nn.BatchNorm2d)
    assert block.conv1.weight.shape == (64,3,3,3)

def test_basic_block_forward():
    """Test the forward pass of BasicBlock without needing to downsample"""
    block = BasicBlock(32, 32)
    x = torch.rand(1, 32, 64, 64)  # Input tensor with batch size 1
    out = block(x)
    assert out.shape == (1, 32, 64, 64)  # Expect the same spatial dimensions

def test_basic_block_downsampling():
    """Test the forward pass of BasicBlock with downsampling"""

    # Since downsampling is handeled by ResNet, we need to manually pass it in
    downsample = nn.Sequential(nn.Conv2d(3, 32, kernel_size=1, stride=1, bias=False))
    block = BasicBlock(3, 32, downsample=downsample)
    x = torch.rand(1, 3, 64, 64)  # Input tensor with batch size 1
    out = block(x)
    assert out.shape == (1, 32, 64, 64)

def test_basic_block_requires_downsampling():
    """
    Test that BasicBlock raises an AssertionError
    when downsampling is required but not provided.
    """
    inplanes, planes, stride = 64, 128, 2  # Example values that require downsampling

    # Expect an AssertionError because downsample is None
    # and the conditions for downsampling are met
    with pytest.raises(ValueError, match="Expected inplanes == planes*expansion for downsample=None"):
        BasicBlock(inplanes, planes, stride=stride)

def test_resnet_initialization():
    """
    Test the initialization of entire ResNet model
    """
    model = resnet18()
    assert isinstance(model, ResNet)

    #Basic shape checks
    assert model.conv1.in_channels == 3
    assert model.conv1.out_channels == 32
    assert model.fc.out_features == 10

    # Check number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum(np.prod(p.size()) for p in model_parameters)
    assert params ==  2797610
