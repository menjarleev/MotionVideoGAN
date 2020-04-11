import torch as t
from torch import nn

class Conv(t.nn.Module):
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
        avg_out = t.mean(x, dim=1, keepdim=True)
        max_out, _ = t.max(x, dim=1, keepdim=True)
        x = t.cat([avg_out, max_out], dim=1)
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
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding='reflect'):
        super(RCBAM, self).__init__()
        self.conv = Conv(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.channel_attention = ChannelAttention(out_channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, feature):
        x = self.conv(feature)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return feature + x

class RCLSTMBAM(t.nn.Module):
    def __init__(self, n_channel, input_size, kernel_size=3, stride=1, padding='reflect'):
        super(RCLSTMBAM, self).__init__()

        self.feature_size = input_size
        self.n_channel = n_channel
        self.compression = Conv(n_channel * 2, n_channel * 4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.out = Conv(n_channel, n_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, original_input, cell_state, hidden_state):
        combined_input = t.cat([original_input, hidden_state], 1)
        combined_conv = self.compression(combined_input)
        cc_i, cc_f, cc_o, cc_g = t.split(combined_conv, self.n_channel, dim = 1)
        i = t.sigmoid(cc_i)
        f = t.sigmoid(cc_f)
        o = t.sigmoid(cc_o)
        g = t.tanh(cc_g)
        c_next = f * cell_state + i * g
        h_next = o * t.tanh(c_next)
        # activation
        x = t.relu(self.out(h_next))
        out = original_input + x
        return out, c_next, h_next

    def init_hidden(self, batch_size):
        return (t.zeros(batch_size, self.n_channel, self.feature_size, self.feature_size),
                t.zeros(batch_size, self.n_channel, self.feature_size, self.feature_size))

class ConvLSTMBAM(t.nn.Module):
    def __init__(self, n_channel, input_size, kernel_size=3, stride=1, padding='reflect'):
        super(ConvLSTMBAM, self).__init__()

        self.feature_size = input_size
        self.n_channel = n_channel
        self.compression = Conv(n_channel * 2, n_channel * 4, kernel_size=kernel_size, stride=stride, padding=padding)
        self.out = Conv(n_channel, n_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, original_input, cell_state, hidden_state):
        combined_input = t.cat([original_input, hidden_state], 1)
        combined_conv = self.compression(combined_input)
        cc_i, cc_f, cc_o, cc_g = t.split(combined_conv, self.n_channel, dim = 1)
        i = t.sigmoid(cc_i)
        f = t.sigmoid(cc_f)
        o = t.sigmoid(cc_o)
        g = t.tanh(cc_g)
        c_next = f * cell_state + i * g
        h_next = o * t.tanh(c_next)
        # activation
        x = t.relu(self.out(h_next))
        out = original_input + x
        return out, c_next, h_next

    def init_hidden(self, batch_size):
        return (t.zeros(batch_size, self.n_channel, self.feature_size[0], self.feature_size[1]),
                t.zeros(batch_size, self.n_channel, self.feature_size[0], self.feature_size[1]))
