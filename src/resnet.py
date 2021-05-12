import torch
from torch import nn


class BasicRSB(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicRSB, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.downsample = (stride > 1 or in_channels != out_channels)
        if self.downsample:
            self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bn3 = nn.BatchNorm1d(out_channels)

        self.fc1 = nn.Linear(out_channels, out_channels // 4)
        self.bn4 = nn.BatchNorm1d(out_channels // 4)
        self.fc2 = nn.Linear(out_channels // 4, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        residual = inputs
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))

        abs_mean = torch.mean(torch.abs(out.permute(0, 2, 1)), dim=1)
        scales = self.relu(self.bn4(self.fc1(abs_mean)))
        scales = self.sigmoid(self.fc2(scales))
        thres = torch.mul(abs_mean, scales).unsqueeze(-1)

        out = torch.mul(torch.sign(out), torch.max(torch.abs(out) - thres, torch.zeros_like(out)))

        if self.downsample:
            residual = self.bn3(self.conv3(residual))
        out += residual
        out = self.relu(out)
        return out


class BottleneckRSB(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckRSB, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU()

        self.downsample = (stride > 1 or in_channels != out_channels * self.expansion)
        if self.downsample:
            self.conv4 = nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bn4 = nn.BatchNorm1d(out_channels * self.expansion)

        self.fc1 = nn.Linear(out_channels * self.expansion, (out_channels * self.expansion) // 4)
        self.bn5 = nn.BatchNorm1d((out_channels * self.expansion) // 4)
        self.fc2 = nn.Linear((out_channels * self.expansion) // 4, out_channels * self.expansion)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        residual = inputs

        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        abs_mean = torch.mean(torch.abs(out.permute(0, 2, 1)), dim=1)
        scales = self.relu(self.bn5(self.fc1(abs_mean)))
        scales = self.sigmoid(self.fc2(scales))
        thres = torch.mul(abs_mean, scales).unsqueeze(-1)

        out = torch.mul(torch.sign(out), torch.max(torch.abs(out) - thres, torch.zeros_like(out)))

        if self.downsample:
            residual = self.bn4(self.conv4(residual))

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.downsample = (stride > 1 or in_channels != out_channels)
        if self.downsample:
            self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, inputs):
        residual = inputs
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.bn3(self.conv3(residual))
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU()

        self.downsample = (stride > 1 or in_channels != out_channels * self.expansion)
        if self.downsample:
            self.conv4 = nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bn4 = nn.BatchNorm1d(out_channels * self.expansion)

    def forward(self, inputs):
        residual = inputs

        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample:
            residual = self.bn4(self.conv4(residual))

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, emb_size, layers, num_classes=50):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(emb_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, nblock, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for i in range(1, nblock):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        inputs = self.relu(self.bn1(self.conv1(inputs)))
        inputs = self.maxpool(inputs)

        inputs = self.layer1(inputs)
        inputs = self.layer2(inputs)
        inputs = self.layer3(inputs)
        inputs = self.layer4(inputs)

        inputs = self.avgpool(inputs)
        inputs = torch.flatten(inputs, 1)
        out = self.fc(inputs)
        return out


if __name__ == '__main__':
    resnet18 = ResNet(Bottleneck, 4, [3, 4, 6, 3])

    num_conv = 0
    for name, param in resnet18.named_parameters():
        print('{}: {}'.format(name, param.size()))
        if name.find('conv') != -1:
            num_conv += 1
    print('Number of convolution layers: {}'.format(num_conv))

    inputs = torch.randn([5, 200, 4])
    out = resnet18(inputs.permute(0, 2, 1))
    print(out.size())
