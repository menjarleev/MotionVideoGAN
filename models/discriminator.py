import torch as t
from torch import nn
from .modules import Conv, ConvLSTMBAM
import numpy as np
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
        self.n_scale = opt.n_scale
        self.resize = get_downsample_layer()
        self.n_layers = opt.d_n_layers
        for i in range(self.n_scale):
            if i == 0:
                from_rgb = [Conv(input_dim, pref, kernel_size=7), nn.LeakyReLU(negative_slope,True)]
                downsample = [Conv(pref, ndf_list[i], kernel_size=3, stride=2, padding=padding), norm_layer(ndf_list[i]), nn.LeakyReLU(negative_slope,True)]
                setattr(self, 'rgb_from_scale_' + str(i), nn.Sequential(*from_rgb))
                setattr(self, 'downsampler_from_scale_' + str(i), nn.Sequential(*downsample))
            else:
                ndf_list += [min(ndf_list[-1] * 2, 512)]
                from_rgb = [Conv(input_dim, ndf_list[-2], kernel_size=7), nn.LeakyReLU(negative_slope,True)]
                downsample = [Conv(ndf_list[-2], ndf_list[-1], kernel_size=3, stride=2), norm_layer(ndf_list[-1]), nn.LeakyReLU(negative_slope,True)]
                setattr(self, 'rgb_from_scale_' + str(i), nn.Sequential(*from_rgb))
                setattr(self, 'downsampler_from_scale_' + str(i), nn.Sequential(*downsample))
        bottom_layer = []
        for i in range(self.n_layers):
            ndf_list += [min(ndf_list[-1] * 2, 512)]
            bottom_layer += nn.Sequential(Conv(ndf_list[-2], ndf_list[-1], kernel_size=3, stride=2, padding=padding), norm_layer(ndf_list[-1]), nn.LeakyReLU(negative_slope, True))
        self.bottom_layer = nn.ModuleList(bottom_layer)
        self.out = Conv(ndf_list[-1], 1, kernel_size=1, stride=1, padding=padding)

    def forward(self, input, scale, w=1):
        out = [input]
        if scale == self.n_scale - 1:
            assert w == 1

        for i in range(scale, self.n_scale):
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
        for i in range(len(self.bottom_layer)):
            out += [self.bottom_layer[i](out[-1])]
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
        self.n_scale = opt.n_scale
        self.resize = get_downsample_layer()
        for i in range(self.n_scale):
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
        if scale == self.n_scale - 1:
            assert w == 1

        for i in range(scale, self.n_scale):
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

class WeightedMultiscaleDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WeightedMultiscaleDiscriminator, self).__init__()
        self.num_D = opt.num_D
        self.n_scale = opt.n_scale
        netD = []
        for i in range(self.num_D):
            netD += [ImageDiscriminator(opt)]
        self.netD = nn.ModuleList(netD)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input, scale=0, weight=1.0):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            result.append(self.netD[i](input, scale, weight))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, opt, n_frames_D):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = opt.num_D
        self.getIntermFeat = not opt.no_ganFeat
        self.n_layers = opt.d_n_layers
        self.n_frames_D = n_frames_D
        ndf_max = 64
        for i in range(self.num_D):
            netD = NLayerDiscriminator(opt, min(ndf_max, opt.ndf * (2 ** (self.num_D - 1 - i))), n_frames_D)
            if self.getIntermFeat:
                for j in range(self.n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, opt, ndf, n_frames_D):
        super(NLayerDiscriminator, self).__init__()
        ndf = ndf
        norm_layer = get_norm_layer(opt.norm_layer)
        input_nc = (opt.input_dim + opt.output_dim) * n_frames_D
        self.getIntermFeat = not opt.no_ganFeat
        self.n_layers = opt.d_n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, self.n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if self.getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)










