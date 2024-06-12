''' ECBSR arch '''
import torch
from torch import nn
from torch.nn import functional as F
from typing import OrderedDict
import math

class SeqConv3x3(nn.Module):
    ''' SeqConv3x3 '''
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super().__init__()

        self.type = seq_type
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.mid_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            conv1 = torch.nn.Conv2d(self.mid_planes,
                                    self.out_planes,
                                    kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.out_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes, ))
            self.bias = nn.Parameter(bias)
            self.mask = torch.zeros((self.out_planes, 1, 3, 3),
                                    dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.out_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes, ))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            self.mask = torch.zeros((self.out_planes, 1, 3, 3),
                                    dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(inp_planes,
                                    self.out_planes,
                                    kernel_size=1,
                                    padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes, ))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            self.mask = torch.zeros((self.out_planes, 1, 3, 3),
                                    dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        ''' forward '''
        if self.type == 'conv1x1-conv3x3':
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            y1 = F.conv2d(input=y0,
                          weight=self.scale * self.mask,
                          bias=self.bias,
                          stride=1,
                          groups=self.out_planes)
        return y1


    def rep_params(self):
        ''' rep_params '''
        device = self.k0.get_device()
        if device < 0:
            device = None
        if self.type == 'conv1x1-conv3x3':
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            RB = torch.ones(1, self.mid_planes, 3, 3,
                            device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1, ) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3),
                             device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            RB = torch.ones(1, self.out_planes, 3, 3,
                            device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1, ) + b1
        return RK, RB


class ECB(nn.Module):
    ''' ECB block '''
    def __init__(self,
                 inp_planes,
                 out_planes,
                 depth_multiplier,
                 act_type='prelu',
                 deploy=False):
        super().__init__()

        self.deploy = deploy
        self.in_channels = inp_planes
        self.out_channels = out_planes

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=inp_planes, out_channels=out_planes,
                                         kernel_size=(3, 3), stride=1,
                                         padding=1, dilation=1, groups=1, bias=True,
                                         padding_mode='zeros')
        else:
            self.conv3x3 = torch.nn.Conv2d(inp_planes,
                                           out_planes,
                                           kernel_size=3,
                                           padding=1)
            self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', inp_planes,
                                          out_planes, depth_multiplier)
            self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', inp_planes, out_planes,
                                          -1)
            self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', inp_planes, out_planes,
                                          -1)
            self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', inp_planes,
                                          out_planes, -1)

        self.act = self.get_activation(act_type, out_planes)

    @staticmethod
    def get_activation(act_type, channels):
        ''' get activation '''
        if act_type == 'prelu':
            return nn.PReLU(num_parameters=channels)
        if act_type == 'relu':
            return nn.ReLU(inplace=True)
        if act_type == 'rrelu':
            return nn.RReLU(lower=-0.05, upper=0.05)
        if act_type == 'softplus':
            return nn.Softplus()
        if act_type == 'linear':
            return nn.Identity()
        if act_type == 'silu':
            return nn.SiLU()
        if act_type == 'gelu':
            return nn.GELU()
        raise ValueError('The type of activation if not support!')

    def forward(self, inputs):
        ''' forward '''
        if (self.deploy):
            y = self.rbr_reparam(inputs)

        else:
            y = self.conv3x3(inputs) + \
                self.conv1x1_3x3(inputs) + \
                self.conv1x1_sbx(inputs) + \
                self.conv1x1_sby(inputs) + \
                self.conv1x1_lpl(inputs)
        y = self.act(y)
        return y


    def switch_to_deploy(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                                     stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1_3x3')
        self.__delattr__('conv1x1_sbx')
        self.__delattr__('conv1x1_sby')
        self.__delattr__('conv1x1_lpl')
        self.deploy = True


    def get_equivalent_kernel_bias(self):
        ''' rep params '''
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        RK, RB = (K0 + K1 + K2 + K3 + K4), (B0 + B1 + B2 + B3 + B4)
        return RK, RB


class NRB(nn.Module): #####################inference
    def __init__(self, n_feats, deploy=True):
        super(NRB, self).__init__()
        self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

    def forward(self, x):
        out = self.rep_conv(x)
        return out


class NRBACT(nn.Module):
    def __init__(self, feature_nums, deploy=False):
        super(NRBACT, self).__init__()
        self.nrb = NRB(n_feats=feature_nums, deploy=deploy)
        self.act = nn.GELU()

    def forward(self, inputs):
        outputs = self.nrb(inputs)
        outputs = self.act(outputs)

        return outputs


class ANUNet(nn.Module):
    ''' ANUNet '''
    def __init__(self, num_in_ch=3, num_out_ch=3, upscale=4, num_block=3, num_feat=28,
                 act_type='gelu', deploy=False):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.upscale = upscale

        backbone = []
        backbone += [
            ECB(4 * num_in_ch, num_feat, depth_multiplier=2.0, act_type=act_type, deploy=deploy)
        ]
        for _ in range(num_block):
            backbone += [
                NRBACT(num_feat, deploy=deploy)
            ]
        backbone += [
            ECB(num_feat,
                4 * num_out_ch * (upscale**2),
                depth_multiplier=2.0,
                act_type='linear', deploy=deploy)
        ]

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(upscale)

        self.down = nn.PixelUnshuffle(2)
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        ''' forward '''
        if self.num_in_ch > 1:
            shortcut = torch.repeat_interleave(x, self.upscale * self.upscale, dim=1)
        else:
            shortcut = x
        x = self.down(x)
        y = self.backbone(x)
        y = self.up(y)
        y = y + shortcut
        y = self.upsampler(y)
        return y



def LeRTSR(scale):

    model = ANUNet(upscale=scale, deploy=True)

    return model

