import torch
import torch.nn as nn
from torchvision.models.resnet import resnet34
#from torchvision.models import resnet50
from .DeiT import deit_small_patch16_224 as deit
from .DeiT import deit_base_patch16_224 as deit_base
from .DeiT import deit_base_patch16_384 as deit_base_384
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class EEM(nn.Module):
    def __init__(self, in_chans1, in_chans2,
        **kwargs, ):
        super().__init__()
        
        self.conv1_1 = nn.Conv2d(in_chans1, 512, kernel_size=3, stride=1, padding=1 )
        self.conv1_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1 )
        self.relu1 = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(256)
        
        self.conv2_1 = nn.Conv2d(in_chans2, 128, kernel_size=3, stride=1, padding=1 )
        self.conv2_2 = nn.Conv2d(128, in_chans2, kernel_size=3, stride=1, padding=1 )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.conv1_1(x1)
        x1 = self.conv1_2(x1)
        x1 = self.relu1(x1)
        x1 = self.bn(x1)
        #print(x1.shape)
        x2 = self.conv2_1(x2)
        x2 = self.conv2_2(x2)
        x2 = self.relu2(x2)
        
        fuse = x1+x2
        
        return fuse

class FFN(nn.Module):#FFN(ch_1, ch_int)
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        # max_out = max_out.view(-1, ch_2, 1, 1)
        # print(x.shape)#x.shape为[2,256,14,14]
        # print(max_out.shape)#max_out.shape为[2,256,1,1]
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_1, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_1, 1, bn=True, relu=False)
        self.W = Conv(ch_1, ch_1, 3, bn=True, relu=True)

        self.FFN = FFN(ch_1, ch_1)
        self.CBAM= CBAMLayer(ch_1)

        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1+ch_1, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

        
    def forward(self, g, x, ch_1):
        #print(x.shape)
        #print(g.shape)
        # bilinear pooling
        W_g = self.W_g(g)
        #print(W_g.shape)
        W_x = self.W_x(x)#W_g与W_x输出一致torch.Size([2, 256, 12, 16])
        #print(W_x.shape)
        bp = self.avg_pool(W_g + W_x)#torch.Size([2, 256, 1, 1])
        #print(bp.shape)
        bp = bp.view(-1, ch_1)#torch.Size([b, 256])
        #print(bp.shape)
        bp = self.FFN(bp)#torch.Size([b, 256])
        #print(bp.shape)
        bp = self.softmax(bp)#torch.Size([b, 256])
        #print(bp.shape)
        bp = bp.reshape(-1, ch_1, 1, 1)#torch.Size([b, 256, 1, 1])
        #print(bp.shape)

        # spatial attention for cnn branch
        g = W_g
        g = self.CBAM(g)#torch.Size([b, 256, 12, 16])
        #print(g.shape)

        # channel attetion for transformer branch
        x = W_x
        x = self.CBAM(x)#torch.Size([b, 384, 12, 16])
        #x = x.conv2d(x.shape[1],ch_1)
        #print(x.shape)

        fuse = self.residual(torch.cat([x * bp, g * bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


class TransFuse_S(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, normal_init=True, pretrained=False):
        super(TransFuse_S, self).__init__()

        self.resnet = resnet34()
        if pretrained:
            self.resnet.load_state_dict(torch.load('pretrained/resnet34-333f7ec4.pth'))
        self.resnet.fc = nn.Identity()
        self.resnet.layer4 = nn.Identity()

        self.transformer = deit(pretrained=pretrained)

        self.up1 = Up(in_ch1=384, out_ch=128)
        self.up2 = Up(128, 64)

        self.final_x = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )

        self.final_1 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )

        self.final_2 = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
            )
        
        self.extract = EEM(in_chans1=384, in_chans2=256)#b,256,12,16

        self.up_c = BiFusion_block(ch_1=256, ch_2=384, ch_out=256, drop_rate=drop_rate/2)#b,256,12,16

        self.up_c_1_1 = BiFusion_block(ch_1=128, ch_2=128, ch_out=128, drop_rate=drop_rate/2)
        self.up_c_1_2 = Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True)

        self.up_c_2_1 = BiFusion_block(ch_1=64, ch_2=64, ch_out=64, drop_rate=drop_rate/2)
        self.up_c_2_2 = Up(128, 64, 64, attn=True)

        self.drop = nn.Dropout2d(drop_rate)

        if normal_init:
            self.init_weights()

    def forward(self, imgs, labels=None):
        # bottom-up path
        x_b = self.transformer(imgs)#torch.Size([b, 192, 384])
        #print(x_b.shape)
        x_b = torch.transpose(x_b, 1, 2)
        #print(x_b.shape)
        x_b = x_b.view(x_b.shape[0], -1, 12, 16)#torch.Size([b, 384, 12, 16])
        #print(x_b.shape)
        x_b = self.drop(x_b)

        x_b_1 = self.up1(x_b)#torch.Size([b, 128, 24, 32])尺寸扩大2倍，通道减少一半
        #print(x_b_1.shape)
        x_b_1 = self.drop(x_b_1)

        x_b_2 = self.up2(x_b_1)  # transformer pred supervise here
        #print(x_b_2.shape)#torch.Size([2, 64, 48, 64])
        x_b_2 = self.drop(x_b_2)

        # top-down path
        x_u = self.resnet.conv1(imgs)
        x_u = self.resnet.bn1(x_u)
        x_u = self.resnet.relu(x_u)
        x_u = self.resnet.maxpool(x_u)
        #print(x_u.shape)#torch.Size([2, 64, 48, 64])

        x_u_2 = self.resnet.layer1(x_u)
        x_u_2 = self.drop(x_u_2)
        #print(x_u_2.shape)#torch.Size([2, 64, 48, 64])

        x_u_1 = self.resnet.layer2(x_u_2)
        x_u_1 = self.drop(x_u_1)
        #print(x_u_1.shape)#torch.Size([2, 128, 24, 32])

        x_u = self.resnet.layer3(x_u_1)
        x_u = self.drop(x_u)
        #print(x_u.shape)#torch.Size([2, 256, 12, 16])

        # joint path
        x_e = self.extract(x_b, x_u)
        x_c = self.up_c(x_u, x_b, ch_1=256)
        x_c = x_c+x_e

        x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1, ch_1=128)
        x_c_1 = self.up_c_1_2(x_c, x_c_1_1)

        x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2, ch_1=64)
        x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1) # joint predict low supervise here

        # decoder part
        map_x = F.interpolate(self.final_x(x_c), scale_factor=16, mode='bilinear', align_corners=True)
        map_1 = F.interpolate(self.final_1(x_b_2), scale_factor=4, mode='bilinear', align_corners=True)
        map_2 = F.interpolate(self.final_2(x_c_2), scale_factor=4, mode='bilinear', align_corners=True)
        return map_x, map_1, map_2

    def init_weights(self):
        self.up1.apply(init_weights)
        self.up2.apply(init_weights)
        self.final_x.apply(init_weights)
        self.final_1.apply(init_weights)
        self.final_2.apply(init_weights)
        self.up_c.apply(init_weights)
        self.up_c_1_1.apply(init_weights)
        self.up_c_1_2.apply(init_weights)
        self.up_c_2_1.apply(init_weights)
        self.up_c_2_2.apply(init_weights)

def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1+in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x)+self.identity(x))


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x