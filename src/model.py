import torch.nn as nn

from src.blocks import CNN
from src.resnet import ResNet, BasicRSB, BasicBlock, Bottleneck, BottleneckRSB


class CNNClassifier(nn.Module):
    def __init__(self, in_channels):
        super(CNNClassifier, self).__init__()
        self.cnn = CNN(in_channels, 50, [3, 4, 5])
        self.fc = nn.Linear(3 * 50, 50)

    def forward(self, inputs):
        return self.fc(self.cnn(inputs))


class ResNetClassifier(nn.Module):
    depth_map = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }

    def __init__(self, in_channels, shrink=False, depth=18):
        super(ResNetClassifier, self).__init__()
        assert depth in self.depth_map
        if shrink:
            blocks = [BasicRSB, BottleneckRSB]
        else:
            blocks = [BasicBlock, Bottleneck]
        self.resnet = ResNet(blocks[int(depth >= 50)], in_channels, self.depth_map[depth])

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        return self.resnet(inputs)
