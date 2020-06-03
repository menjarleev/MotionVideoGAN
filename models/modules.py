import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from collections import OrderedDict
import re

class Conv(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding='reflect', stride=1, bias=False):
        super(Conv, self).__init__()
        pad = 0
        has_pad = False
        if isinstance(kernel_size, int) and kernel_size // 2 > 0:
            has_pad = True
            pad = kernel_size // 2

        if has_pad:
            self.conv = nn.Sequential(nn.ReflectionPad2d(pad),
                                      nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=0,
                                                bias=bias))
        else:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=0,
                                  bias=bias)

    def forward(self, x):
        return self.conv(x)


class ChannelAttentionModule(torch.nn.Module):
    def __init__(self, hi, wi):
        super(ChannelAttentionModule, self).__init__()
        kernel = (hi, wi)
        self.k_conv = Conv(in_channel=1, out_channel=1, kernel_size=kernel)
        self.q_conv = Conv(in_channel=1, out_channel=1, kernel_size=kernel)
        self.v_conv = Conv(in_channel=1, out_channel=1, kernel_size=kernel)

    def forward(self, input):
        b, c, hi, wi = input.shape
        input = input.view(-1, 1, hi, wi)
        k = self.k_conv(input)
        q = self.q_conv(input)
        v = self.v_conv(input)
        k = k.view(b, c, -1).permute(0, 2, 1)
        q = q.view(b, c, -1)
        v = v.view(b, c, -1)
        w = q @ k
        w = F.softmax(w, dim=-1)
        attn = w @ v
        attn = attn.view(b, c, 1, 1)
        return attn


class SpatialAttentionModule(torch.nn.Module):
    def __init__(self, in_channel, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.q_conv = Conv(in_channel=in_channel, out_channel=1, kernel_size=1)
        self.k_conv = Conv(in_channel=in_channel, out_channel=1, kernel_size=1)
        self.v_conv = Conv(in_channel=in_channel, out_channel=1, kernel_size=1)
        self.transform = Conv(in_channel=1, out_channel=1, kernel_size=kernel_size)

    def forward(self, input):
        b, c, hi, wi = input.shape
        q = self.q_conv(input)
        k = self.k_conv(input)
        v = self.v_conv(input)
        q = q.view(b, 1, -1).permute(0, 2, 1)
        k = k.view(b, 1, -1)
        w = q @ k
        w = F.softmax(w, -1)
        v = v.view(b, 1, -1).permute(0, 2, 1)
        attn = w @ v
        attn = attn.view(b, 1, hi, wi)
        attn = self.transform(attn)
        return attn

class FusionChannelAttention(nn.Module):
    def __init__(self, in_planes, hi, wi, ratio=4, fusion=True):
        super(FusionChannelAttention, self).__init__()
        self.fusion = fusion
        if self.fusion:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.attn_pool = ChannelAttentionModule(hi=hi, wi=wi)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, phi=1):
        merge = 0
        if self.fusion:
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
            merge = avg_out + max_out
        feat = self.attn_pool(x)
        attn_out = self.fc2(self.relu1(self.fc1(feat)))
        return self.sigmoid(phi * attn_out + (1 - phi) * merge)


class FusionSpatialAttention(nn.Module):
    def __init__(self, n_channel, kernel_size=7, fusion=True):
        super(FusionSpatialAttention, self).__init__()
        self.fusion = fusion
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        if self.fusion:
            self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.attn_pool = SpatialAttentionModule(in_channel=n_channel, kernel_size=kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, phi=1):
        merge = 0
        if self.fusion:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            merge = torch.cat([avg_out, max_out], dim=1)
        attn_out = self.attn_pool(x)
        merge = self.conv1(merge)
        x = phi * attn_out + (1 - phi) * merge
        return self.sigmoid(x)

class FusionResACM(nn.Module):
    def __init__(self, in_channel, hi, wi, norm_layer, activation=nn.ReLU(True), kernel_size=3, stride=1, padding='reflect', fusion=True):
        super(FusionResACM, self).__init__()
        self.conv1 = Conv(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.channel1 = FusionChannelAttention(in_channel, hi, wi, fusion=fusion)
        self.spatial1 = FusionSpatialAttention(in_channel, fusion=fusion)
        self.norm1 = norm_layer(in_channel)
        self.act = activation
        self.conv2 = Conv(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.channel2 = FusionChannelAttention(in_channel, hi, wi, fusion=fusion)
        self.spatial2 = FusionSpatialAttention(in_channel, fusion=fusion)
        self.norm2 = norm_layer(in_channel)

    def forward(self, feat, phi=1):
        x = self.conv1(feat)
        x = self.channel1(x, phi) * x
        x = self.spatial1(x, phi) * x
        x = self.act(self.norm1(x))
        x = self.conv2(x)
        x = self.channel1(x, phi) * x
        x = self.spatial1(x, phi) * x
        x = self.norm2(x)
        return x + feat

# class UniSCConvBlock(torch.nn.Module):
#     def __init__(self, in_channel, feat_hi, feat_wi, kernel_size=3, ratio=16):
#         super(UniSCConvBlock, self).__init__()
#         self.conv = Conv(in_channel, in_channel, 3)
#         self.channelAttention = ChannelAttentionModule(feat_hi, feat_wi, kernel_size=kernel_size)
#         self.spatialAttention = SpatialAttentionModule(in_channel=in_channel, ratio=ratio)
#
#     def forward(self, input):
#         x = self.conv(input)
#         x = self.channelAttention(x)
#         x = self.spatialAttention(x)
#         return x
#
# class UniSCResBlock(torch.nn.Module):
#     def __init__(self, in_channel, feat_hi, feat_wi, kernel_size=3, ratio=16, norm_layer=nn.BatchNorm2d, actv=nn.ReLU(True)):
#         super(UniSCResBlock, self).__init__()
#         self.UniSCBlock1 = UniSCConvBlock(in_channel, feat_hi, feat_wi, kernel_size, ratio)
#         self.act = nn.Sequential(norm_layer(in_channel), actv)
#         self.UniSCBlock2 = UniSCConvBlock(in_channel, feat_hi, feat_wi, kernel_size, ratio)
#         self.norm = norm_layer(in_channel)
#
#     def forward(self, input):
#         x = self.UniSCBlock1(input)
#         x = self.act(x)
#         x = self.UniSCBlock2(x)
#         x = self.norm(x)
#         x = x + input
#         return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding='reflect'):
        super(CBAM, self).__init__()
        self.conv = Conv(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.channel_attention = ChannelAttention(out_channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x

class AdaptiveChannelAttention(nn.Module):
    def __init__(self, in_planes, hi, wi, ratio=4, fusion=True):
        super(AdaptiveChannelAttention, self).__init__()
        self.fusion = fusion
        if self.fusion:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
        kernel = (hi, wi)
        self.attn_pool = Conv(1, 1, kernel_size=kernel)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, phi=1):
        out = 0
        if self.fusion:
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
            out = avg_out + max_out
        b, c, hi, wi = x.shape
        feat = x.view(-1, 1, hi, wi)
        feat = self.attn_pool(feat).view(b, c, 1, 1)
        attn_out = self.fc2(self.relu1(self.fc1(feat)))
        return self.sigmoid(phi * attn_out + (1 - phi) * out)

class AdaptiveSpatialAttention(nn.Module):
    def __init__(self, n_channel, kernel_size=7, fusion=True):
        super(AdaptiveSpatialAttention, self).__init__()
        self.fusion = fusion
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.attn_pool = nn.Conv2d(n_channel, 2, kernel_size=1)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, phi=1):
        merge = 0
        if self.fusion:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            merge = torch.cat([avg_out, max_out], dim=1)
        attn_out = self.attn_pool(x)
        x = self.conv1(phi * attn_out + (1 - phi) * merge)
        return self.sigmoid(x)


class AdaptiveRCBAM(nn.Module):
    def __init__(self, in_channel, hi, wi, norm_layer, activation=nn.ReLU(True), kernel_size=3, stride=1, padding='reflect', fusion=True):
        super(AdaptiveRCBAM, self).__init__()
        self.conv1 = Conv(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.channel1 = AdaptiveChannelAttention(in_channel, hi, wi, fusion=fusion)
        self.spatial1 = AdaptiveSpatialAttention(in_channel, fusion=fusion)
        self.norm1 = norm_layer(in_channel)
        self.act = activation
        self.conv2 = Conv(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.channel2 = AdaptiveChannelAttention(in_channel, hi, wi, fusion=fusion)
        self.spatial2 = AdaptiveSpatialAttention(in_channel, fusion=fusion)
        self.norm2 = norm_layer(in_channel)

    def forward(self, feat, phi=1):
        x = self.conv1(feat)
        x = self.channel1(x, phi) * x
        x = self.spatial1(x, phi) * x
        x = self.act(self.norm1(x))
        x = self.conv2(x)
        x = self.channel1(x, phi) * x
        x = self.spatial1(x, phi) * x
        x = self.norm2(x)
        return x + feat

class RCBAM(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer, activation=nn.ReLU(True), kernel_size=3, stride=1, padding='reflect'):
        super(RCBAM, self).__init__()
        self.conv1 = Conv(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.channel1 = ChannelAttention(out_channel)
        self.spatial1 = SpatialAttention()
        self.norm1 = norm_layer(out_channel)
        self.act =  activation
        self.conv2 = Conv(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.channel2 = ChannelAttention(out_channel)
        self.spatial2 = SpatialAttention()
        self.norm2 = norm_layer(out_channel)



    def forward(self, feature):
        x = self.conv1(feature)
        x = self.channel1(x) * x
        x = self.spatial1(x) * x
        x = self.act(self.norm1(x))
        x = self.conv2(x)
        x = self.channel1(x) * x
        x = self.spatial1(x) * x
        x = self.norm2(x)
        return x + feature

class RConv(nn.Module):
    def __init__(self, n_channel, ratio=4, padding='reflect', activation=nn.ReLU(True)):
        super(RConv, self).__init__()
        self.n_channel = n_channel
        in_channel = n_channel * 2
        self.net = nn.Sequential(Conv(in_channel, in_channel // ratio, padding=padding), activation, Conv(in_channel // ratio, in_channel, padding=padding))
        self.hidden_state = None

    def init_state(self, feature_size, batch_size=1, gpu_id=0):
        self.hidden_state = torch.zeros(batch_size, self.n_channel, feature_size[0], feature_size[1]).cuda(gpu_id)
        return self.hidden_state

    def detach_state(self):
        self.hidden_state = self.hidden_state.detach()

    def forward(self, feature):
        x = torch.cat([feature, self.hidden_state], dim=1)
        x = self.net(x)
        out, self.hidden_state = torch.split(x, self.n_channel, dim=1)
        return out + feature



class RCLSTMBAM(nn.Module):
    def __init__(self, n_channel, kernel_size=3, stride=1, padding='reflect'):
        super(RCLSTMBAM, self).__init__()
        self.n_channel = n_channel
        self.compression = Conv(n_channel * 2, n_channel * 4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.out = Conv(n_channel, n_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.cell_state = None
        self.hidden_state = None

    def forward(self, original_input):
        combined_input = torch.cat([original_input, self.hidden_state], 1)
        combined_conv = self.compression(combined_input)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.n_channel, dim = 1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        self.cell_state = f * self.cell_state + i * g
        self.hidden_state = o * torch.tanh(self.cell_state)
        # activation
        x = torch.relu(self.out(self.hidden_state))
        out = original_input + x
        return out

    def init_state(self, feature_size, batch_size=1, gpu_id=0):
        self.hidden_state = torch.zeros(batch_size, self.n_channel, feature_size, feature_size).cuda(gpu_id)
        self.cell_state = torch.zeros(batch_size, self.n_channel, feature_size, feature_size).cuda(gpu_id)
        return [self.hidden_state, self.cell_state]

    def detach_state(self):
        self.hidden_state = self.hidden_state.detach()
        self.cell_state = self.cell_state.detach()

class ResnetBlock(nn.Module):
    def __init__(self, dim,  norm_layer, activation=nn.ReLU(True), use_dropout=False, padding_type='reflect'):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ConvLSTMBAM(nn.Module):
    def __init__(self, n_channel, kernel_size=3, stride=1, padding='reflect'):
        super(ConvLSTMBAM, self).__init__()

        self.n_channel = n_channel
        self.compression = Conv(n_channel * 2, n_channel * 4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.out = Conv(n_channel, n_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.cell_state = None
        self.hidden_state = None

    def forward(self, original_input):
        combined_input = torch.cat([original_input, self.hidden_state], 1)
        combined_conv = self.compression(combined_input)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.n_channel, dim = 1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        self.cell_state = f * self.cell_state + i * g
        self.hidden_state = o * torch.tanh(self.cell_state)
        # activation
        x = torch.relu(self.out(self.hidden_state))
        out = original_input + x
        return out

    def init_state(self, feature_size, batch_size=1, gpu_id=0):
        self.hidden_state = torch.zeros(batch_size, self.n_channel, feature_size[0], feature_size[1]).cuda(gpu_id)
        self.cell_state = torch.zeros(batch_size, self.n_channel, feature_size[0], feature_size[1]).cuda(gpu_id)
        return [self.hidden_state, self.cell_state]

    def detach_state(self):
        self.hidden_state = self.hidden_state.detach()
        self.cell_state = self.cell_state.detach()

class BranchResBlock(nn.Module):
    def __init__(self, n_channel, kernel_size=3, stride=1, padding='reflect', activation=nn.ReLU(True), norm_layer=nn.BatchNorm2d, ratio=4):
        super(BranchResBlock, self).__init__()
        self.n_channel = n_channel
        self.conv = Conv(n_channel, 3 * n_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bg = nn.Sequential(norm_layer(n_channel), activation, Conv(n_channel, n_channel, kernel_size=kernel_size, stride=stride, padding=padding))
        self.fg = nn.Sequential(norm_layer(n_channel), activation, Conv(n_channel, n_channel, kernel_size=kernel_size, stride=stride, padding=padding))
        self.mask  = nn.Sequential(norm_layer(n_channel), activation, Conv(n_channel, n_channel, kernel_size=kernel_size, stride=stride, padding=padding))
        self.sigmoid = nn.Sigmoid()
        self.channel_attention = ChannelAttention(n_channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, input):
        x = self.conv(input)
        mask, bg, fg = torch.split(x, self.n_channel, dim=1)
        mask = self.mask(mask)
        mask = self.channel_attention(mask) * mask
        mask = self.spatial_attention(mask) * mask
        mask = self.sigmoid(mask)
        x = bg * mask + (1 - mask) * fg
        return x + input

class AttentionConvLSTM(nn.Module):
    def __init__(self, n_channel, kernel_size=3, stride=1, padding='reflect', activation=nn.ReLU(True), norm_layer=nn.BatchNorm2d, ratio=4):
        super(AttentionConvLSTM, self).__init__()
        self.n_channel = n_channel
        self.align_hidden = Conv(n_channel * 2, n_channel, kernel_size=kernel_size,stride=stride, padding=padding)
        self.align_cell = Conv(n_channel * 2, n_channel, kernel_size=kernel_size,stride=stride, padding=padding)
        self.compression = Conv(n_channel * 2, n_channel * 4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.out = nn.Sequential(Conv(n_channel, n_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                                 norm_layer(n_channel), activation)
        self.cell_state = None
        self.hidden_state = None

    def forward(self, original_input):
        pre_hidden = torch.cat([original_input, self.hidden_state], 1)
        pre_cell = torch.cat([original_input, self.cell_state], 1)
        hidden_in = self.align_hidden(pre_hidden)
        cell_in = self.align_cell(pre_cell)
        combined_input = torch.cat([hidden_in, original_input], 1)
        combined_conv = self.compression(combined_input)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.n_channel, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        self.cell_state = f * cell_in + i * g
        self.hidden_state = o * torch.tanh(self.cell_state)
        # activation
        x = self.out(self.hidden_state)
        out = original_input + x
        return out

    def init_state(self, feature_size, batch_size=1, gpu_id=0):
        self.hidden_state = torch.zeros(batch_size, self.n_channel, feature_size[0], feature_size[1]).cuda(gpu_id)
        self.cell_state = torch.zeros(batch_size, self.n_channel, feature_size[0], feature_size[1]).cuda(gpu_id)
        return [self.hidden_state, self.cell_state]

    def detach_state(self):
        self.hidden_state = self.hidden_state.detach()
        self.cell_state = self.cell_state.detach()

class AlignedConvLSTM(nn.Module):
    def __init__(self, n_channel, kernel_size=3, stride=1, padding='reflect', activation=nn.ReLU(True), norm_layer=nn.BatchNorm2d, ratio=4):
        super(AlignedConvLSTM, self).__init__()
        self.n_channel = n_channel
        self.align_hidden = Conv(n_channel * 2, n_channel, kernel_size=kernel_size,stride=stride, padding=padding)
        self.align_cell = Conv(n_channel * 2, n_channel, kernel_size=kernel_size,stride=stride, padding=padding)
        self.compression = Conv(n_channel * 2, n_channel * 4, kernel_size=kernel_size, stride=stride, padding=padding)
        out = [Conv(n_channel, n_channel, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm_layer is not None:
            out += [norm_layer(n_channel)]
        out += [activation]
        self.out = nn.Sequential(*out)
        self.cell_state = None
        self.hidden_state = None

    def forward(self, original_input):
        pre_hidden = torch.cat([original_input, self.hidden_state], 1)
        pre_cell = torch.cat([original_input, self.cell_state], 1)
        hidden_in = self.align_hidden(pre_hidden)
        cell_in = self.align_cell(pre_cell)
        combined_input = torch.cat([hidden_in, original_input], 1)
        combined_conv = self.compression(combined_input)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.n_channel, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        self.cell_state = f * cell_in + i * g
        self.hidden_state = o * torch.tanh(self.cell_state)
        # activation
        x = self.out(self.hidden_state)
        out = original_input + x
        return out

    def init_state(self, feature_size, batch_size=1, gpu_id=0):
        self.hidden_state = torch.zeros(batch_size, self.n_channel, feature_size[0], feature_size[1]).cuda(gpu_id)
        self.cell_state = torch.zeros(batch_size, self.n_channel, feature_size[0], feature_size[1]).cuda(gpu_id)
        return [self.hidden_state, self.cell_state]

    def detach_state(self):
        self.hidden_state = self.hidden_state.detach()
        self.cell_state = self.cell_state.detach()

#################### openpose ####################
##################################################
def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))

class bodypose_model(nn.Module):
    def __init__(self):
        super(bodypose_model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',\
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',\
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',\
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict([
                      ('conv1_1', [3, 64, 3, 1, 1]),
                      ('conv1_2', [64, 64, 3, 1, 1]),
                      ('pool1_stage1', [2, 2, 0]),
                      ('conv2_1', [64, 128, 3, 1, 1]),
                      ('conv2_2', [128, 128, 3, 1, 1]),
                      ('pool2_stage1', [2, 2, 0]),
                      ('conv3_1', [128, 256, 3, 1, 1]),
                      ('conv3_2', [256, 256, 3, 1, 1]),
                      ('conv3_3', [256, 256, 3, 1, 1]),
                      ('conv3_4', [256, 256, 3, 1, 1]),
                      ('pool3_stage1', [2, 2, 0]),
                      ('conv4_1', [256, 512, 3, 1, 1]),
                      ('conv4_2', [512, 512, 3, 1, 1]),
                      ('conv4_3_CPM', [512, 256, 3, 1, 1]),
                      ('conv4_4_CPM', [256, 128, 3, 1, 1])
                  ])


        # Stage 1
        block1_1 = OrderedDict([
                        ('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
                        ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
                        ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])
                    ])

        block1_2 = OrderedDict([
                        ('conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
                        ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
                        ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
                        ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
                        ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])
                    ])
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2

        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict([
                    ('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
                    ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])
                ])

            blocks['block%d_2' % i] = OrderedDict([
                    ('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
                    ('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])
                ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']

        self.model1_2 = blocks['block1_2']
        self.model2_2 = blocks['block2_2']
        self.model3_2 = blocks['block3_2']
        self.model4_2 = blocks['block4_2']
        self.model5_2 = blocks['block5_2']
        self.model6_2 = blocks['block6_2']


    def forward(self, x):
        out_list = []
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)
        out_list += [out2]

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)
        out_list += [out3]

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)
        out_list += [out4]

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)
        out_list += [out5]

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)
        out_list += [out6]

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)
        out_list += [out6_1]
        out_list += [out6_2]

        return out_list

class handpose_model(nn.Module):
    def __init__(self):
        super(handpose_model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3',\
                          'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6']
        # stage 1
        block1_0 = OrderedDict([
                ('conv1_1', [3, 64, 3, 1, 1]),
                ('conv1_2', [64, 64, 3, 1, 1]),
                ('pool1_stage1', [2, 2, 0]),
                ('conv2_1', [64, 128, 3, 1, 1]),
                ('conv2_2', [128, 128, 3, 1, 1]),
                ('pool2_stage1', [2, 2, 0]),
                ('conv3_1', [128, 256, 3, 1, 1]),
                ('conv3_2', [256, 256, 3, 1, 1]),
                ('conv3_3', [256, 256, 3, 1, 1]),
                ('conv3_4', [256, 256, 3, 1, 1]),
                ('pool3_stage1', [2, 2, 0]),
                ('conv4_1', [256, 512, 3, 1, 1]),
                ('conv4_2', [512, 512, 3, 1, 1]),
                ('conv4_3', [512, 512, 3, 1, 1]),
                ('conv4_4', [512, 512, 3, 1, 1]),
                ('conv5_1', [512, 512, 3, 1, 1]),
                ('conv5_2', [512, 512, 3, 1, 1]),
                ('conv5_3_CPM', [512, 128, 3, 1, 1])
            ])

        block1_1 = OrderedDict([
            ('conv6_1_CPM', [128, 512, 1, 1, 0]),
            ('conv6_2_CPM', [512, 22, 1, 1, 0])
        ])

        blocks = {}
        blocks['block1_0'] = block1_0
        blocks['block1_1'] = block1_1

        # stage 2-6
        for i in range(2, 7):
            blocks['block%d' % i] = OrderedDict([
                    ('Mconv1_stage%d' % i, [150, 128, 7, 1, 3]),
                    ('Mconv2_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv3_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv4_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv5_stage%d' % i, [128, 128, 7, 1, 3]),
                    ('Mconv6_stage%d' % i, [128, 128, 1, 1, 0]),
                    ('Mconv7_stage%d' % i, [128, 22, 1, 1, 0])
                ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6

class ScaleDown(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, bias=False):
        super(ScaleDown, self).__init__()
        self.bottleneck = Conv(in_channel, out_channel, kernel_size=kernel_size, stride=stride, bias=bias)
        self.downsampler = nn.AvgPool2d(kernel_size=2, stride=2, count_include_pad=False)
        self.upsampler = nn.Upsample(scale_factor=2)
        self.conv = Conv(out_channel * 2, out_channel, kernel_size=3, stride=1, bias=bias)

    def forward(self, x):
        x_1 = x
        x_2 = self.downsampler(x_1)
        x_3 = self.downsampler(x_2)
        x_1 = self.bottleneck(x_1)
        x_2 = self.bottleneck(x_2)
        x_3 = self.bottleneck(x_3)
        x_1 = self.downsampler(x_1)
        x_3 = self.upsampler(x_3)
        mean_feat = (x_1 + x_2 + x_3) / 3
        max_feat = torch.max(torch.max(x_1, x_2), x_3)
        feat = torch.cat([mean_feat, max_feat], dim=1)
        out = self.conv(feat)
        return out

class ScaleUp(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, bias=False):
        self.upsampler = nn.UPsample()

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
