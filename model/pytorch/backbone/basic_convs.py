from torch import nn

def conv3x3(
    in_planes: int, out_planes: int,
    stride: int=1, padding: int=1, groups: int=1, dilation: int=1,
    bias: bool=False
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels=in_planes, out_channels=out_planes, kernel_size=3,
        stride=stride, padding=padding, groups=groups, dilation=dilation,
        bias=bias
    )

def conv1x1(in_planes: int, out_planes: int, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels=in_planes, out_channels=out_planes,
        kernel_size=1, stride=stride, bias=False
    )

