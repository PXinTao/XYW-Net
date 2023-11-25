import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
# import utils

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encode = encode()
        self.decode = decode_rcf()

    def forward(self, x):
        end_point = self.encode(x)
        x = self.decode(end_point)
        return x
class decode_rcf(nn.Module):
    def __init__(self):
        super(decode_rcf, self).__init__()
        self.f43 = Refine_block2_1(in_channel=(120, 120), out_channel=60, factor=2)
        self.f32 = Refine_block2_1(in_channel=(60, 60), out_channel=30, factor=2)
        self.f21 = Refine_block2_1(in_channel=(30, 30), out_channel=24, factor=2)
        self.f = nn.Conv2d(24,1,kernel_size=1,padding=0)
    def forward(self,x):

        s3 = self.f43(x[2], x[3])
        s2 = self.f32(x[1], s3)
        s1 = self.f21(x[0], s2)
        x = self.f(s1)
        return x.sigmoid()
class p2d(nn.Module):
    def __init__(self,channel):
        super(p2d, self).__init__()
        self.p2d_v = Conv2d(pdc_func='p2d', in_channels=channel, out_channels=channel, kernel_size=(3, 1), padding=(1, 0))
        self.p2d_h = Conv2d(pdc_func='p2d', in_channels=channel, out_channels=1, kernel_size=(1, 3), padding=(0, 1))

        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        p2d_v = self.p2d_v(x)
        p2d_v = self.relu(p2d_v)


        p2d_h = self.p2d_h(p2d_v)

        return p2d_h


def crop(x, shape):
    _, _, h, w = shape
    _, _, _h, _w = x.shape
    p_h = (_h - h) // 2 + 1
    p_w = (_w - w) // 2 + 1
    return x[:, :, p_h:p_h+h, p_w:p_w+w]
class encode(nn.Module):
    def __init__(self):
        super(encode, self).__init__()
        self.s1 = s1()
        self.s2 = s2()
        self.s3 = s3()
        self.s4 = s4()

    def forward(self, x):
        s1 = self.s1(x)

        s2 = self.s2(s1)

        s3 = self.s3(s2)

        s4 = self.s4(s3)

        return s1, s2, s3, s4


class decode(nn.Module):
    def __init__(self,channel=30):
        super(decode, self).__init__()

        self.conv1 = nn.Conv2d(channel, 1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(channel*2, 1, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(channel*4, 1, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(channel*4, 1, kernel_size=1, padding=0)
        self.conv6 = nn.Conv2d(4, 1, kernel_size=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 3:
                    torch.nn.init.constant_(m.weight, 0.33)  # 初始为常数

                else:
                    torch.nn.init.normal_(m.weight, 0, 1e-2)  # 正态分布

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input, otherends=None):

        _, _, h0, w0 = input[0].size()
        s1 = self.conv1(input[0])
        s2 = self.conv2(input[1])
        s3 = self.conv3(input[2])
        s4 = self.conv4(input[3])
        # s3 = F.interpolate(input=s3, size=(h0//2, w0//2), mode='bilinear', align_corners=False)

        # s3 = F.interpolate(input=s3, size=(h0, w0), mode='bilinear', align_corners=False)


        # cat = self.conv6(torch.cat([s1, s2, s3], dim=1))
        # s3 *= cat.sigmoid()
        s2 = F.interpolate(input=s2, size=(h0, w0), mode='bilinear', align_corners=False)
        s3 = F.interpolate(input=s3, size=(h0, w0), mode='bilinear', align_corners=False)
        s4 = F.interpolate(input=s4, size=(h0, w0), mode='bilinear', align_corners=False)
        return [self.conv6(torch.cat([s1, s2, s3,s4], dim=1)).sigmoid(), s1.sigmoid(), s2.sigmoid(), s3.sigmoid(),s4.sigmoid()]


class CDCM_CSAM_decoder(nn.Module):
    def __init__(self):
        super(CDCM_CSAM_decoder, self).__init__()
        # 膨胀卷积，提高感受野
        self.stage1_CDCM = CDCM(in_channels=60, out_channels=24)
        self.stage2_CDCM = CDCM(in_channels=120, out_channels=24)
        self.stage3_CDCM = CDCM(in_channels=240, out_channels=24)

        # 注意力机制
        self.stage_CSAM = CSAM(channels=24)  # 输入输出channels一致

        self.conv1_1 = nn.Conv2d(in_channels=24, out_channels=1, kernel_size=1, padding=0)
        self.fusion = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        _, _, H, W = x[0].size()
        # 膨胀卷积，提高感受野
        # print(x[0].shape)
        cd1 = self.stage1_CDCM(x[0])

        cd2 = self.stage2_CDCM(x[1])
        cd3 = self.stage3_CDCM(x[2])

        # 空间注意力机制
        cs1 = self.stage_CSAM(cd1)
        cs2 = self.stage_CSAM(cd2)
        cs3 = self.stage_CSAM(cd3)

        # 压缩通道
        y1 = self.conv1_1(cs1)
        y2 = self.conv1_1(cs2)
        y3 = self.conv1_1(cs3)

        # 双线性插值上采样
        y1 = F.interpolate(y1, [H, W], mode="bilinear", align_corners=False)
        y2 = F.interpolate(y2, [H, W], mode="bilinear", align_corners=False)
        y3 = F.interpolate(y3, [H, W], mode="bilinear", align_corners=False)

        y = self.fusion(torch.cat([y1, y2, y3], dim=1)).sigmoid()

        return y, y1.sigmoid(), y2.sigmoid(), y3.sigmoid()


class s1(nn.Module):
    def __init__(self,channel=30):
        super(s1, self).__init__()

        self.conv1 = nn.Conv2d(3, channel, kernel_size=7, padding=6,dilation=2)
        self.xyw1_1 = XYW_S(channel, channel)
        self.xyw1_2 = XYW(channel, channel)
        self.xyw1_3 = XYW_E(channel, channel)

        self.relu = nn.ReLU()

    def forward(self, x):
        temp = self.relu(self.conv1(x))
        xc, yc, w = self.xyw1_1(temp)
        xc, yc, w = self.xyw1_2(xc, yc, w)
        xyw1_3 = self.xyw1_3(xc, yc, w)

        return xyw1_3 + temp
        # return xyw1_3


class s2(nn.Module):
    def __init__(self,channel=60):
        super(s2, self).__init__()
        self.xyw2_1 = XYW_S(channel//2, channel, stride=2)
        self.xyw2_2 = XYW(channel, channel)
        self.xyw2_3 = XYW_E(channel, channel)
        self.shortcut = nn.Conv2d(in_channels=channel//2, out_channels=channel, kernel_size=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.pool(x)
        xc, yc, w = self.xyw2_1(x)
        xc, yc, w = self.xyw2_2(xc, yc, w)
        xyw2_3 = self.xyw2_3(xc, yc, w)
        shortcut = self.shortcut(x)

        return xyw2_3 + shortcut
        # return xyw2_3
class s3(nn.Module):
    def __init__(self,channel=120):
        super(s3, self).__init__()
        self.xyw3_1 = XYW_S(channel//2, channel, stride=2)
        self.xyw3_2 = XYW(channel, channel)
        self.xyw3_3 = XYW_E(channel, channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut = nn.Conv2d(in_channels=channel // 2, out_channels=channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.pool(x)
        shortcut = self.shortcut(x)
        xc, yc, w= self.xyw3_1(x)
        xc, yc, w= self.xyw3_2(xc, yc, w)
        xyw3_3 = self.xyw3_3(xc, yc, w)

        return xyw3_3+shortcut

        # return xyw3_3
class s4(nn.Module):
    def __init__(self,channel=120):
        super(s4, self).__init__()
        self.xyw4_1 = XYW_S(channel, channel, stride=2)
        self.xyw4_2 = XYW(channel, channel)
        self.xyw4_3 = XYW_E(channel, channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut = nn.Conv2d(in_channels=channel , out_channels=channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.pool(x)
        shortcut = self.shortcut(x)
        xc, yc, w = self.xyw4_1(x)
        xc, yc, w = self.xyw4_2(xc, yc, w)
        xyw4_3 = self.xyw4_3(xc, yc, w)

        return xyw4_3+shortcut

        # return xyw4_3

class XYW(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(XYW, self).__init__()
        self.y_c = Yc1x1(inchannel, outchannel)
        self.x_c = Xc1x1(inchannel, outchannel)
        self.w = W(inchannel, outchannel)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, xc, yc, w):

        xc = self.x_c(xc)
        yc = self.y_c(yc)
        w = self.w(w)

        return xc, yc, w

class XYW_S(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(XYW_S, self).__init__()
        self.stride = stride
        self.y_c = Yc1x1(inchannel , outchannel)
        self.x_c = Xc1x1(inchannel , outchannel)
        self.w = W(inchannel, outchannel)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):

        xc = self.x_c(x)
        yc = self.y_c(x)
        w = self.w(x)
        return xc, yc, w
class XYW_E(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(XYW_E, self).__init__()
        self.y_c = Yc1x1(inchannel , outchannel)
        self.x_c = Xc1x1(inchannel , outchannel)
        self.w = W(inchannel, outchannel)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, xc, yc, w):
        xc = self.x_c(xc)
        yc = self.y_c(yc)
        w = self.w(w)

        return xc+yc+w
class Xc1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Xc1x1, self).__init__()

        self.Xcenter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Xcenter_relu = nn.ReLU(inplace=True)

        self.Xsurround = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv1_1 = nn.Conv2d(out_channels,out_channels,kernel_size=1)
        self.Xsurround_relu = nn.ReLU(inplace=True)


        self.in_channels = in_channels

    def forward(self, input):
        xcenter = self.Xcenter_relu(self.Xcenter(input))
        xsurround = self.Xsurround_relu(self.Xsurround(input))
        xsurround = self.conv1_1(xsurround)

        x = xsurround - xcenter


        return x


class Yc1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Yc1x1, self).__init__()
        self.Ycenter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Ycenter_relu = nn.ReLU(inplace=True)

        self.Ysurround = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=4, dilation=2, groups=in_channels)
        self.conv1_1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.Ysurround_relu = nn.ReLU(inplace=True)



    def forward(self, input):
        ycenter = self.Ycenter_relu(self.Ycenter(input))
        ysurround = self.Ysurround_relu(self.Ysurround(input))
        ysurround = self.conv1_1(ysurround)
        y = ysurround - ycenter


        return y


class W(nn.Module):
    def __init__(self, inchannel, outchannel,stride=1):
        super(W, self).__init__()

        self.h = nn.Conv2d(inchannel, inchannel, kernel_size=(1, 3), padding=(0, 1),groups=inchannel)
        self.v = nn.Conv2d(inchannel, inchannel, kernel_size=(3, 1), padding=(1, 0),groups=inchannel)

        self.convh_1 = nn.Conv2d(in_channels= inchannel, out_channels=inchannel, kernel_size=1, padding=0,bias=False)
        self.convv_1 = nn.Conv2d(in_channels= inchannel, out_channels=outchannel, kernel_size=1, padding=0,bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):


        h = self.relu(self.h(x))
        h = self.convh_1(h)

        v = self.relu(self.v(h))
        v = self.convv_1(v)

        return v

class adap_conv(nn.Module):
    def __init__(self, in_channels, out_channels,kz=3,pd=1):
        super(adap_conv, self).__init__()
        self.conv = nn.Sequential(*[Conv2d(pdc_func='2sd', in_channels=in_channels, out_channels=out_channels, kernel_size=kz, padding=pd),
                                    nn.InstanceNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        return x


class Refine_block2_1(nn.Module):
    def __init__(self, in_channel, out_channel, factor, require_grad=False):
        super(Refine_block2_1, self).__init__()
        self.pre_conv1 = adap_conv(in_channel[0], out_channel,kz=3,pd=1)
        self.pre_conv2 = adap_conv(in_channel[1], out_channel,kz=3,pd=1)
        self.factor = factor

        self.deconv_weight = nn.Parameter(bilinear_upsample_weights(factor, out_channel), requires_grad=require_grad)

    def forward(self, *input):
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])

        x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor/2),
                                output_padding=(x1.size(2) - x2.size(2)*self.factor, x1.size(3) - x2.size(3)*self.factor))
        return x1 + x2

def createPDCFunc(PDC_type):  # 创建像素差卷积函数
    assert PDC_type in ['cv', 'cd', 'ad', 'rd', 'sd','p2d','2sd','2cd'], 'unknown PDC type: %s' % str(PDC_type)

    if PDC_type == 'cv':  # 采用香草卷积
        return F.conv2d

    if PDC_type == 'cd':  # CPDC,基于中心差的像素差卷积
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'  # 卷积核膨胀最大为2，这个2在哪用？没找到
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'
            # print('a',weights[0,0,:,:])
            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            # shape = weights.shape
            # if weights.is_cuda:
            #     buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 3).fill_(0)
            # else:
            #     buffer = torch.zeros(shape[0], shape[1], 3 * 3)
            # # 对于一个卷积核,拉成一条直线,方便索引
            # buffer = weights.clone()
            # buffer[:, :, buffer.shape[2] // 2, buffer.shape[3] // 2] = weights[:, :, weights.shape[2] // 2,
            #                                                            weights.shape[3] // 2] * 2
            # weights = buffer.view(shape)
            # print(weights[0,0,:,:])
            # 把3x3卷积核weights求sum变成1x1的卷积核weights_c,即∑wi
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            # 一个点乘以∑wi得到新的feature map的一个点,Ycj = Xj*∑wi
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            # 正常的香草卷积，Yj=∑wi*xi,新的一个点是上一层9个点计算出来，得到feature map
            return y - yc
            # 公式7,对于CPDC结果的一个点j，有∑wi*xi - Xj*∑wi,即相当于把差分卷积变成两个普通卷积之差

        return func
    elif PDC_type == 'sd':  # CPDC,基于周围差的像素差卷积
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 3).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 3 * 3)
            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引
            buffer = weights.clone()
            # print(buffer)
            # buffer = weights
            # 1 2 3
            # 4 5 6   ---------->  [ 1 2 3 4 5 6 7 8 9 ]
            # 7 8 9

            buffer[:, :, [4]] = weights[:, :, [4]] - 2 * weights[:, :, [0, 1, 2, 3, 5, 6, 7, 8]].sum(dim=-1, keepdims=True)

            weights = buffer.view(shape)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    elif PDC_type == 'ad':  # 基于角度差的PDC
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引
            # 1 2 3
            # 4 5 6   ---------->  [ 1 2 3 4 5 6 7 8 9 ]
            # 7 8 9
            # y = w1*(x1-x2) + w2*(x2-x3) + ... + w4*(x4-x1)
            #   = (w1 - w4)*x1 + (w2 - w1)*x2 + ... + (w9 - w6)*x9 公式8
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  # clock-wise卷积核权重相减
            # 公式8实现，转变成权重相减得到新卷积核，再用这个卷积核去进行传统的卷积运算
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    elif PDC_type == 'rd':  # 基于径向差分的PDC
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)  # 拉成一条直线,方便索引
            # 依然是利用权重按径向相减变成新权重构造新卷积核的思想，实现公式9
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            # w1 w3 w5 w11 w15 w21 w23 w25 为正
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            # w7 w8 w9 w12 w14 w17 w18 w19 为负
            buffer[:, :, 12] = 0  # 卷积核中心
            buffer = buffer.view(shape[0], shape[1], 5, 5)  # 再转变成5x5的shape
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    elif PDC_type == 'p2d':  # CPDC,基于周围差的像素差卷积

        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 or weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            # assert padding == dilation, 'padding for ad_conv set wrong'
            # print(weights[0][0])
            shape = weights.shape
            # if weights.is_cuda:
            #     buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 1).fill_(0)
            # else:
            #     buffer = torch.zeros(shape[0], shape[1], 3 * 1)
            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引
            buffer = weights.clone()
            # print(buffer)
            # buffer = weights
            # 1 2 3
            # 4 5 6   ---------->  [ 1 2 3 4 5 6 7 8 9 ]
            # 7 8 9

            buffer[:, :, [0, 2]] = buffer[:, :, [0, 2]] - weights[:, :, [1]]
            # buffer[:, :, [2]] = weights[:, :, [2]] - weights[:, :, [1]]
            buffer[:, :, [1]] = 0

            weights = buffer.view(shape)
            # print(weights[0][0])
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    elif PDC_type == '2cd':  # CPDC,基于中心差的像素差卷积
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 or weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            # assert padding == dilation, 'padding for ad_conv set wrong'
            # print('0',weights[0][0])
            shape = weights.shape
            # if weights.is_cuda:
            #     buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 1).fill_(0)
            # else:
            #     buffer = torch.zeros(shape[0], shape[1], 3 * 1)
            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引
            buffer = weights.clone()
            # print(buffer)
            # buffer = weights
            # 1 2 3
            # 4 5 6   ---------->  [ 1 2 3 4 5 6 7 8 9 ]
            # 7 8 9

            buffer[:, :, [0, 1, 2, 3, 5, 6, 7, 8]] = buffer[:, :, [0, 1, 2, 3, 5, 6, 7, 8]] + buffer[:, :,
                                                                                              [2, 7, 8, 5, 3, 0, 1,
                                                                                               6]] - 2 * buffer[:, :,
                                                                                                         [4]]
            # buffer[:, :, [2]] = weights[:, :, [2]] - weights[:, :, [1]]
            buffer[:, :, [4]] = 0

            weights = buffer.view(shape)
            # print(weights[0][0])
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    elif PDC_type == '2sd':  # CPDC,基于周围差的像素差卷积
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 or weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            # assert padding == dilation, 'padding for ad_conv set wrong'
            # print('0', weights[0][0])
            shape = weights.shape
            # if weights.is_cuda:
            #     buffer = torch.cuda.FloatTensor(shape[0], shape[1], 3 * 1).fill_(0)
            # else:
            #     buffer = torch.zeros(shape[0], shape[1], 3 * 1)
            weights = weights.view(shape[0], shape[1], -1)  # 对于一个卷积核,拉成一条直线,方便索引
            buffer = weights.clone()
            # print(buffer)
            # buffer = weights
            # 1 2 3
            # 4 5 6   ---------->  [ 1 2 3 4 5 6 7 8 9 ]
            # 7 8 9

            buffer[:, :, [0, 1, 2, 3, 5, 6, 7, 8]] = buffer[:, :, [0, 1, 2, 3, 5, 6, 7, 8]] + buffer[:, :,
                                                                                              [2, 7, 8, 5, 3, 0, 1,
                                                                                               6]] - 2 * buffer[:, :,
                                                                                                         [1, 4, 5, 4, 4,
                                                                                                          3, 4, 7]]
            # buffer[:, :, [2]] = weights[:, :, [2]] - weights[:, :, [1]]
            buffer[:, :, [4]] = 0
            weights = buffer.view(shape)
            # print(weights[0][0])
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y

        return func
    else:
        print('unknown PDC type: %s' % str(PDC_type))  # 正常来说走不到这里
        return None


class Conv2d(nn.Module):  # 把之前创建的卷积函数包装成torch卷积api相同的格式
    def __init__(self, pdc_func, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False):
        """
        :param pdc_func: 卷积函数
        """
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:  # depth wise卷积要求通道要能被分组数整除
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation  # 用于控制空洞卷积，默认为1，卷积核尺寸不膨胀
        self.groups = groups
        # print(self.kernel_size)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc_func = createPDCFunc(pdc_func)

    def reset_parameters(self):
        # 凯明初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        #               输入      卷积核权重     偏置                                卷积核膨胀（用于空洞卷积）
        return self.pdc_func(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



class x_off(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(x_off, self).__init__()
        self.x_c = nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1)
        self.x_s = nn.Conv2d(inchannel, outchannel, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_c = self.relu(self.x_c(x))
        x_s = self.relu(self.x_s(x))
        x_off = x_c - x_s
        return x_off


class y_off(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(y_off, self).__init__()
        self.y_c = nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1)
        self.y_s = Conv2d(inchannel, outchannel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y_c = self.relu(self.y_c(x))
        y_s = self.relu(self.y_s(x))
        y_off = y_c - y_s
        return y_off


class y_on(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(y_on, self).__init__()
        self.y_c = nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=2, dilation=2)
        self.y_s = Conv2d(inchannel, outchannel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y_c = self.relu(self.y_c(x))
        y_s = self.relu(self.y_s(x))
        y_on = y_s - y_c
        return y_on




class CSAM(nn.Module):
    # 紧凑型空间注意力机制
    # 把c个channels的feature map通过卷积变成channel=1的weights map,再与原feature map相乘
    # 输出channels等于输入channels
    def __init__(self, channels):
        super(CSAM, self).__init__()
        self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(in_channels=channels, out_channels=4, kernel_size=1, padding=0)
        self.conv3_3 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1_1.bias, 0)

    def forward(self, x):
        y = self.relu(x)
        y = self.conv1_1(y)
        y = self.conv3_3(y)
        y = self.sigmoid(y)
        return x * y


class CDCM(nn.Module):
    # 紧凑膨胀卷积
    # 膨胀卷积,不改变feature map尺寸的前提下，提高感受野
    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()

        self.relu1 = nn.ReLU()
        # 4路同时，以获取不同大小的感受野的feature map,但尺寸一致
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = 2 * factor - factor % 2
    weights = np.zeros((number_of_classes,
                        number_of_classes,
                        filter_size,
                        filter_size,), dtype=np.float32)
    upsample_kernel = upsample_filt(filter_size)

    for i in range(number_of_classes):
        weights[i, i, :, :] = upsample_kernel
    return torch.Tensor(weights)
if __name__ == '__main__':
    x = torch.randn(1,3,512,512)
    net = Net()
    print(net(x)[0].shape)
