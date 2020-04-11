import torch as t
from torch import nn
from .modules import Conv, ConvLSTMBAM
from .model_helper import get_downsample_layer, get_norm_layer


class ImageDiscriminator(nn.Module):

    def __init__(self, opt):
        super(ImageDiscriminator, self).__init__()
        negative_slope = opt.negative_slope
        norm_layer = get_norm_layer(opt.norm_layer)
        input_dim = opt.input_dim + opt.output_dim
        ndf_list = [opt.ndf]
        pref = opt.pref_image_D
        padding = opt.padding_type
        self.n_downsampling = opt.n_downsampling
        self.resize = get_downsample_layer()
        for i in range(self.n_downsampling):
            if i == 0:
                from_rgb = [Conv(input_dim, pref, kernel_size=7, stride=1, padding=padding), norm_layer(pref), nn.LeakyReLU(negative_slope,True)]
                downsample = [Conv(pref, ndf_list[i], kernel_size=3, stride=2, padding=padding), norm_layer(ndf_list[i]), nn.LeakyReLU(negative_slope,True)]
                setattr(self, 'rgb_from_scale_' + str(i), nn.Sequential(*from_rgb))
                setattr(self, 'downsampler_from_scale_' + str(i), nn.Sequential(*downsample))
            else:
                ndf_list += [min(ndf_list[-1] * 2, 512)]
                from_rgb = [Conv(input_dim, ndf_list[-2], kernel_size=7), norm_layer(ndf_list[-2]), nn.LeakyReLU(negative_slope,True)]
                downsample = [Conv(ndf_list[-2], ndf_list[-1], kernel_size=3, stride=2), norm_layer(ndf_list[-1]), nn.LeakyReLU(negative_slope,True)]
                setattr(self, 'rgb_from_scale_' + str(i), nn.Sequential(*from_rgb))
                setattr(self, 'downsampler_from_scale_' + str(i), nn.Sequential(*downsample))
        ndf_list += [min(ndf_list[-1] * 2, 512)]
        self.bottom_layer = nn.Sequential(Conv(ndf_list[-2], ndf_list[-1], kernel_size=3, stride=1, padding=padding), norm_layer(ndf_list[-1]), nn.LeakyReLU(negative_slope, True))
        self.out = Conv(ndf_list[-1], 1, kernel_size=3, stride=1, padding=padding)

    def forward(self, input, scale, w=1):
        out = [input]
        if scale == self.n_downsampling - 1:
            assert w == 1

        for i in range(scale, self.n_downsampling):
            downsampler = getattr(self, 'downsampler_from_scale_' + str(i))
            if i == scale:
                from_rgb = getattr(self, 'rgb_from_scale_' + str(i))
                x = downsampler(from_rgb(out[-1]))
                out += [x]
            elif i == scale + 1 and w != 1:
                from_rgb = getattr(self, 'rgb_from_scale_' + str(i))
                y = from_rgb(self.resize(out[0]))
                x = downsampler(w * out[-1] + (1 - w) * y)
                out += [x]
            else:
                x = downsampler(out[-1])
                out += [x]
        out += [self.bottom_layer(out[-1])]
        out += [self.out(out[-1])]
        # do not return the input
        return out[1:]



class VideoDiscriminator(nn.Module):
    """
    video discriminator, takes n_frames_D [gt/fake, condition] as input, output a patch result
    """

    def __init__(self, opt):
        super(VideoDiscriminator, self).__init__()
        negative_slope = opt.negative_slope
        norm_layer = get_norm_layer(opt.norm_layer)
        input_dim = (opt.input_dim + opt.output_dim) * opt.n_frames_D
        ndf_list = [opt.ndf]
        pref = opt.pref_video_D
        padding = opt.padding_type
        self.n_downsampling = opt.n_downsampling
        self.resize = get_downsample_layer()
        for i in range(self.n_downsampling):
            if i == 0:
                from_rgb = [Conv(input_dim, pref, kernel_size=7, stride=1, padding=padding), norm_layer(pref), nn.LeakyReLU(negative_slope,True)]
                downsample = [Conv(pref, ndf_list[i], kernel_size=3, stride=2, padding=padding), norm_layer(ndf_list[i]), nn.LeakyReLU(negative_slope,True)]
                setattr(self, 'rgb_from_scale_' + str(i), nn.Sequential(*from_rgb))
                setattr(self, 'downsampler_from_scale_' + str(i), nn.Sequential(*downsample))
            else:
                ndf_list += [min(ndf_list[-1] * 2, 512)]
                from_rgb = [Conv(input_dim, ndf_list[-2], kernel_size=7), norm_layer(ndf_list[-2]), nn.LeakyReLU(negative_slope,True)]
                downsample = [Conv(ndf_list[-2], ndf_list[-1], kernel_size=3, stride=2), norm_layer(ndf_list[-1]), nn.LeakyReLU(negative_slope,True)]
                setattr(self, 'rgb_from_scale_' + str(i), nn.Sequential(*from_rgb))
                setattr(self, 'downsampler_from_scale_' + str(i), nn.Sequential(*downsample))
        ndf_list += [min(ndf_list[-1] * 2, 512)]
        self.bottom_layer = nn.Sequential(Conv(ndf_list[-2], ndf_list[-1], kernel_size=3, stride=1, padding=padding), norm_layer(ndf_list[-1]), nn.LeakyReLU(negative_slope, True))
        self.out = Conv(ndf_list[-1], 1, kernel_size=3, stride=1, padding=padding)

    def forward(self, input, scale=0, w=1):
        out = [input]
        if scale == self.n_downsampling - 1:
            assert w == 1

        for i in range(scale, self.n_downsampling):
            downsampler = getattr(self, 'downsampler_from_scale_' + str(i))
            if i == scale:
                from_rgb = getattr(self, 'rgb_from_scale_' + str(i))
                x = downsampler(from_rgb(out[-1]))
                out += [x]
            elif i == scale + 1 and w != 1:
                from_rgb = getattr(self, 'rgb_from_scale_' + str(i))
                y = from_rgb(self.resize(out[0]))
                x = downsampler(w * out[-1] + (1 - w) * y)
                out += [x]
            else:
                x = downsampler(out[-1])
                out += [x]
        out += [self.bottom_layer(out[-1])]
        out += [self.out(out[-1])]
        # do not return the input
        return out[1:]










