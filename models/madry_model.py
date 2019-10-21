'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import pdb
import numpy as np
from collections import OrderedDict
from tf_models.madry_model import Model as MadryTFResNet
import os

class ResidualBlock(nn.Module):
    def __init__(self, in_filter, out_filter, stride,
                  activate_before_residual=False, train_dp=0, test_dp=0):
        super(ResidualBlock, self).__init__()
        self.active_before_residual = activate_before_residual
        self.in_filter = in_filter
        self.out_filter = out_filter
        self.stride = stride
        self.train_dp = train_dp
        self.test_dp = test_dp

        self.init_act = nn.Sequential(OrderedDict([
                        ('init_bn', nn.BatchNorm2d(in_filter)),
                        ('init_relu', nn.LeakyReLU(0.1))
        ]))
        self.transforms = nn.Sequential(OrderedDict([
                        ('conv1', nn.Conv2d(in_filter, out_filter, kernel_size=3, stride=stride, padding=1, bias=False)),
                        ('bn2', nn.BatchNorm2d(out_filter)),
                        ('relu2', nn.LeakyReLU(0.1)),
        ]))
        self.conv2 = nn.Conv2d(out_filter, out_filter, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        if self.active_before_residual:
            x = self.init_act(x)
            orig_x = x
        else:
            orig_x = x
            x = self.init_act(x)

        x = self.transforms(x)

        if self.test_dp > 0 or (self.training and self.train_dp > 0):
            dp = max(self.test_dp, self.train_dp)
            x = F.dropout(x, dp, training=True)

        x = self.conv2(x)
        if self.in_filter != self.out_filter:
            orig_x = F.avg_pool2d(orig_x, self.stride, self.stride)
            pad_dim = (self.out_filter - self.in_filter) // 2
            zeros = torch.zeros(x.size(0), pad_dim, x.size(2), x.size(3)).to(x)
            orig_x = torch.cat([zeros, orig_x, zeros], dim=1)

        return x + orig_x

class MadryResNet(nn.Module):
    def __init__(self, num_classes=10, train_dp=0, test_dp=0, droplayer=0, bdp=0):
        super(MadryResNet, self).__init__()
        self.in_planes = 64

        self.init_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        filters = [16, 160, 320, 640]
        self.activate_before_residual = activate_before_residual
        self.nblk_per_unit = 5

        # self.bn1 = nn.BatchNorm2d(64)


        self.unit_1 = self._make_layer(self.nblk_per_unit, filters[0], filters[1], strides[0], activate_before_residual[0], train_dp=train_dp, test_dp=test_dp)
        self.unit_2 = self._make_layer(self.nblk_per_unit, filters[1], filters[2], strides[1], activate_before_residual[1], train_dp=train_dp, test_dp=test_dp)
        self.unit_3 = self._make_layer(self.nblk_per_unit, filters[2], filters[3], strides[2], activate_before_residual[2], train_dp=train_dp, test_dp=test_dp)

        self.unit_last = nn.Sequential(OrderedDict([
            ("final_bn", nn.BatchNorm2d(filters[3])),
            ("final_relu", nn.LeakyReLU(0.1))
        ]))

        self.linear = nn.Linear(filters[3], num_classes)

        self.test_dp = test_dp

    def set_weight(self, pt_w_name, tf_w_name, sess, transpose=False):
        pt_w = eval('self.%s' % pt_w_name)
        tf_w = sess.run(tf_w_name)
        if len(tf_w.shape) == 4:
            # tf is H x W x in_channels x out_channels
            # pytorch is out_channels x in_channels x H x W
            tf_w = np.transpose(tf_w, (3, 2, 0, 1))
        if transpose:
            # only suitable for transposing weights of fully connected layers
            tf_w = tf_w.T
        for i in range(len(pt_w.size())):
            assert(pt_w.size()[i] == tf_w.shape[i])

        pt_w.data = torch.Tensor(tf_w).to(pt_w)

    def set_bn_weights(self, pt_name, tf_name, sess):
        self.set_weight(pt_name + '.weight', tf_name % 'gamma', sess)
        self.set_weight(pt_name + '.bias', tf_name % 'beta', sess)
        self.set_weight(pt_name + '.running_mean', tf_name % 'moving_mean', sess)
        self.set_weight(pt_name + '.running_var', tf_name % 'moving_variance', sess)


    def load_from_tf(self, model_dir):
        cur_checkpoint = tf.train.latest_checkpoint(model_dir)
        print("Trying to load from {}".format(cur_checkpoint))
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
            saver.restore(sess, cur_checkpoint)
            # get the variables by name with sess.run
            # tt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            self.set_weight('init_conv.weight', "input/init_conv/DW:0", sess)
            for n_unit in range(3):
                for n_blk in range(self.nblk_per_unit):
                    pt_uname = 'unit_%d'%(n_unit+1)
                    uname = 'unit_%d_%d'%(n_unit+1, n_blk)
                    shared_act = self.activate_before_residual[n_unit] and n_blk == 0
                    act_name = 'shared_activation' if shared_act else 'residual_only_activation'

                    # initial BN
                    tf_BN_name = os.path.join(uname, act_name, 'BatchNorm', '%s:0')
                    pt_BN_name = pt_uname + '[%d]'%n_blk + '.init_act[0]'
                    self.set_bn_weights(pt_BN_name, tf_BN_name, sess)

                    # other blocks
                    conv_name = os.path.join(uname, 'sub%d', 'conv%d', 'DW:0')
                    self.set_weight(pt_uname + '[%d]'%n_blk + '.transforms[0].weight', conv_name%(1, 1), sess)

                    tf_BN_name = os.path.join(uname, 'sub2', 'BatchNorm', '%s:0')
                    pt_BN_name = pt_uname + '[%d]'%n_blk + '.transforms[1]'
                    self.set_bn_weights(pt_BN_name, tf_BN_name, sess)

                    self.set_weight(pt_uname + '[%d]'%n_blk + '.conv2.weight', conv_name%(2,2), sess)

            # last BN
            tf_BN_name = 'unit_last/BatchNorm/%s:0'
            pt_BN_name = 'unit_last[0]'
            self.set_bn_weights(pt_BN_name, tf_BN_name, sess)
            self.set_weight('linear.weight', 'logit/DW:0', sess, transpose=True)
            self.set_weight('linear.bias', 'logit/biases:0', sess)
        # ideally add a test here


    def _make_layer(self, n_blks, in_filters, out_filters, init_stride, activate_before_residual,
                    train_dp=0, test_dp=0):
        layers = []
        for n in range(n_blks):
            if n == 0:
                layer = ResidualBlock(in_filters, out_filters, stride=init_stride, activate_before_residual=activate_before_residual,
                                      train_dp=train_dp, test_dp=test_dp)
            else:
                layer = ResidualBlock(out_filters, out_filters, stride=1, activate_before_residual=False,
                                      train_dp=train_dp, test_dp=test_dp)
            layers.append(layer)

        return nn.Sequential(*layers)

    def set_testdp(self, dp):
        for layer in self.unit_1:
            layer.test_dp = dp
        for layer in self.unit_2:
            layer.test_dp = dp
        for layer in self.unit_3:
            layer.test_dp = dp

    def penultimate(self, x):
        out = self.init_conv(x)
        out = self.unit_1(out)
        out = self.unit_2(out)
        out = self.unit_3(out)
        out = self.unit_last(out)
        out = torch.mean(out.view(out.size(0), out.size(1), -1), 2)
        return out

    def reset_last_layer(self):
        self.linear.weight.data.normal_(0, 0.1)
        self.linear.bias.data.zero_()

    def forward(self, x, penu=False, block=False):

        out = self.penultimate(x)
        if penu:
            return out
        out = self.linear(out)
        return out

    def get_penultimate_params_list(self):
        return [param for name, param in self.named_parameters() if 'linear' in name]

if __name__ == "__main__":
    net = MadryResNet()
    net.load_from_tf('/home/chen/sources/cifar10_challenge/models/model_0')