import torch
from torch import nn

class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding='reflect', bias=False):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(nn.ReflectionPad2d(kernel_size // 2),
                                  nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=0,
                                            bias=bias)) if padding is 'reflect' else \
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                      bias=bias)

    def forward(self, x):
        return self.conv(x)

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
