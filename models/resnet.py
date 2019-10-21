'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, train_dp=0, test_dp=0, droplayer=0, bdp=0):
        # if test_dp > 0: will always keep dp there
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.train_dp = train_dp
        self.test_dp = test_dp

        self.droplayer = droplayer

    def forward(self, x):
        action = np.random.binomial(1, self.droplayer)
        if action == 1:
            out = self.shortcut(x)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            if self.test_dp > 0 or (self.training and self.train_dp>0):
                dp = max(self.test_dp, self.train_dp)
                out = F.dropout(out, dp, training=True)
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)

        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, train_dp=0, test_dp=0, droplayer=0, bdp=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.train_dp = train_dp
        self.test_dp = test_dp
        self.bdp = bdp

        self.droplayer = droplayer

    def forward(self, x):
        action = np.random.binomial(1, self.droplayer)
        if action == 1:
            out = self.shortcut(x)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))

            if self.test_dp > 0 or (self.training and self.train_dp>0):
                dp = max(self.test_dp, self.train_dp)
                out = F.dropout(out, dp, training=True)
            if self.bdp > 0:
                # each sample will be applied the same mask
                bdp_mask = torch.bernoulli(
                    self.bdp * torch.ones(1, out.size(1), out.size(2), out.size(3)).to(out.device)) / self.bdp
                out = bdp_mask * out

            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, train_dp=0, test_dp=0, droplayer=0, bdp=0):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        nblks = sum(num_blocks)
        dl_step = droplayer / nblks

        dl_start = 0
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, train_dp=train_dp, test_dp=test_dp,
                                       dl_start=dl_start, dl_step=dl_step, bdp=bdp)

        dl_start += dl_step * num_blocks[0]
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, train_dp=train_dp, test_dp=test_dp,
                                       dl_start=dl_start, dl_step=dl_step, bdp=bdp)

        dl_start += dl_step * num_blocks[1]
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, train_dp=train_dp, test_dp=test_dp,
                                       dl_start=dl_start, dl_step=dl_step, bdp=bdp)

        dl_start += dl_step * num_blocks[2]
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, train_dp=train_dp, test_dp=test_dp,
                                       dl_start=dl_start, dl_step=dl_step, bdp=bdp)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.test_dp = test_dp

    def get_block_feats(self, x):
        feat_list = []

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        feat_list.append(out)

        out = self.layer2(out)
        feat_list.append(out)

        out = self.layer3(out)
        feat_list.append(out)

        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        feat_list.append(out)

        return feat_list

    def set_testdp(self, dp):
        for layer in self.layer1:
            layer.test_dp = dp
        for layer in self.layer2:
            layer.test_dp = dp
        for layer in self.layer3:
            layer.test_dp = dp
        for layer in self.layer4:
            layer.test_dp = dp

    def _make_layer(self, block, planes, num_blocks, stride, train_dp=0, test_dp=0, dl_start=9, dl_step=0, bdp=0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for ns, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, train_dp=train_dp, test_dp=test_dp,
                                droplayer=dl_start+dl_step*ns, bdp=bdp))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def penultimate(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def reset_last_layer(self):
        self.linear.weight.data.normal_(0, 0.1)
        self.linear.bias.data.zero_()

    def forward(self, x, penu=False, block=False):
        if block:
            return self.get_block_feats(x)

        out = self.penultimate(x)
        if penu:
            return out
        out = self.linear(out)
        return out

    def get_penultimate_params_list(self):
        return [param for name, param in self.named_parameters() if 'linear' in name]

def ResNet18(train_dp=0, test_dp=0, droplayer=0, bdp=0):
    return ResNet(BasicBlock, [2,2,2,2], train_dp=train_dp, test_dp=test_dp, droplayer=droplayer, bdp=bdp)

def ResNet34(train_dp=0, test_dp=0, droplayer=0):
    return ResNet(BasicBlock, [3,4,6,3], train_dp=train_dp, test_dp=test_dp, droplayer=droplayer)

def ResNet50(train_dp=0, test_dp=0, droplayer=0, bdp=0):
    return ResNet(Bottleneck, [3,4,6,3], train_dp=train_dp, test_dp=test_dp, droplayer=droplayer, bdp=bdp)

def ResNet101(train_dp=0, test_dp=0, droplayer=0):
    return ResNet(Bottleneck, [3,4,23,3], train_dp=train_dp, test_dp=test_dp, droplayer=droplayer)

def ResNet152(train_dp=0, test_dp=0, droplayer=0):
    return ResNet(Bottleneck, [3,8,36,3], train_dp=train_dp, test_dp=test_dp, droplayer=droplayer)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
