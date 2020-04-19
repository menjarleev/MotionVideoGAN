import torch as t
import torch.nn as nn
import torch.nn.functional as F
from .modules import *
from .model_helper import get_norm_layer, get_upsample_layer, get_downsample_layer

class VideoGenerator(t.nn.Module):

    def __init__(self, opt):
        super(VideoGenerator, self).__init__()
        ngf = opt.ngf
        norm_layer = get_norm_layer(opt.norm_layer)
        input_dim = opt.input_dim * opt.n_frames_G
        output_dim = opt.output_dim
        image_size = opt.image_size
        pref = opt.pref
        padding = opt.padding_type
        n_block = opt.n_block
        ngf_list = [ngf]

        self.n_downsampling = opt.n_downsampling
        self.downsampler = get_downsample_layer()
        self.upsampler = nn.Upsample(scale_factor=2, align_corners=False)
        self.encoder_state = []
        self.decoder_state = []

        # set up encoder
        for i in range(self.n_downsampling):
            if i == 0:
                from_rgb = [Conv(input_dim, pref, kernel_size=7, stride=1, padding=padding), norm_layer(pref), nn.ReLU(True)]
                encoder = [ConvLSTMBAM(pref, image_size), Conv(pref, ngf_list[i], kernel_size=3, stride=2, padding=padding), norm_layer(ngf_list[i]), nn.ReLU(True)]
                setattr(self, 'rgb_from_scale_' + str(i), nn.Sequential(*from_rgb))
                setattr(self, 'encoder_from_scale_' + str(i), nn.ModuleList(encoder))
            else:
                ngf_list += [min(ngf_list[-1] * 2, 512)]
                from_rgb = [Conv(input_dim, ngf_list[-2], kernel_size=7, stride=1, padding=padding), norm_layer(ngf_list[-2]), nn.ReLU(True)]
                encoder = [Conv(ngf_list[-2], ngf_list[-1], kernel_size=3, stride=2, padding=padding), norm_layer(ngf_list[-1]), nn.ReLU(True)]
                setattr(self, 'rgb_from_scale_' + str(i), nn.Sequential(*from_rgb))
                setattr(self, 'encoder_from_scale_' + str(i), nn.Sequential(*encoder))

       # set up res_block
        ngf_list += [ngf_list[-1]]
        res_blocks = []
        for i in range(n_block):
            # res_blocks += [ResnetBlock(ngf_list[-1], norm_layer)]
            res_blocks += [RCBAM(ngf_list[-1], ngf_list[-1], norm_layer)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # set up decoder
        for i in range(self.n_downsampling, 0, -1):
            if i == 1:
                to_rgb = [Conv(pref, output_dim, kernel_size=1, stride=1, padding=padding), nn.Tanh()]
                decoder = [nn.Upsample(scale_factor=2, mode=opt.upsample_type, align_corners=False), Conv(ngf_list[i], ngf_list[i-1], kernel_size=3, stride=1, padding=padding), norm_layer(ngf_list[i-1]), nn.ReLU(True),
                           Conv(ngf_list[i-1], pref, kernel_size=3, stride=1, padding=padding), norm_layer(pref), nn.ReLU(True),
                           ConvLSTMBAM(pref, image_size)]
                # decoder = [nn.ConvTranspose2d(ngf_list[i], ngf_list[i-1], kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf_list[i-1]), nn.ReLU(True),
                #            Conv(ngf_list[i-1], pref, kernel_size=3, stride=1, padding=padding), norm_layer(pref), nn.ReLU(True),
                #            ConvLSTMBAM(pref, image_size)]
                setattr(self, 'rgb_to_scale_' + str(i - 1), nn.Sequential(*to_rgb))
                setattr(self, 'decoder_to_scale_' + str(i - 1), nn.ModuleList(decoder))
            else:
                to_rgb = [Conv(ngf_list[i-1], output_dim, kernel_size=1, stride=1, padding=padding), nn.Tanh()]
                decoder = [nn.Upsample(scale_factor=2, mode=opt.upsample_type, align_corners=False), Conv(ngf_list[i], ngf_list[i - 1], kernel_size=3, stride=1, padding=padding), norm_layer(ngf_list[i - 1]), nn.ReLU(True)]
                # decoder = [nn.ConvTranspose2d(ngf_list[i], ngf_list[i - 1], kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf_list[i - 1]), nn.ReLU(True)]
                setattr(self, 'rgb_to_scale_' + str(i - 1), nn.Sequential(*to_rgb))
                setattr(self, 'decoder_to_scale_' + str(i - 1), nn.Sequential(*decoder))

    def init_state(self, batch_size, gpu_id):
        self.encoder_state = getattr(self,'encoder_from_scale_0')[0].init_hidden(batch_size=batch_size, gpu_id=gpu_id)
        self.decoder_state = getattr(self,'decoder_to_scale_0')[-1].init_hidden(batch_size=batch_size, gpu_id=gpu_id)
        return [self.encoder_state, self.decoder_state]

    def detach_state(self):
        self.encoder_state = [s.detach() for s in self.encoder_state]
        self.decoder_state = [s.detach() for s in self.decoder_state]

    def forward(self, input, scale, w=1):
        if scale == self.n_downsampling - 1:
            assert w == 1

        for i in range(scale, self.n_downsampling):
            encoder = getattr(self, 'encoder_from_scale_' + str(i))
            if i == scale:
                from_rgb = getattr(self,'rgb_from_scale_' + str(i))
                x = from_rgb(input)
                if i == 0:
                    for j in range(len(encoder)):
                        if j == 0:
                            x, c_next, h_next = encoder[j](x, self.encoder_state[0], self.encoder_state[1])
                            self.encoder_state = [c_next, h_next]
                        else:
                            x = encoder[j](x)
                else:
                    x = encoder(x)
            elif i == scale + 1 and w != 1:
                from_rgb = getattr(self, 'rgb_from_scale_' + str(i))
                y = from_rgb(self.downsampler(input))
                x = encoder(w * x + (1 - w) * y)
            else:
                x = encoder(x)

        x = self.res_blocks(x)

        for i in range(self.n_downsampling - 1, scale - 1, -1):
            decoder = getattr(self, 'decoder_to_scale_' + str(i))
            if i == scale + 1 and w != 1:
                to_rgb = getattr(self, 'rgb_to_scale_' + str(i))
                next_decoder = getattr(self, 'decoder_to_scale_' + str(i - 1))
                x = decoder(x)
                y = next_decoder[0](to_rgb(x))
            elif i == scale:
                to_rgb = getattr(self, 'rgb_to_scale_' + str(i))
                if i == 0:
                    for j in range(len(decoder)):
                        if j == len(decoder) - 1:
                            x, c_next, h_next = decoder[j](x, self.decoder_state[0], self.decoder_state[1])
                            self.decoder_state = [c_next, h_next]
                        else:
                            x = decoder[j](x)
                else:
                    x = decoder(x)
                x = to_rgb(x)
                curr = x * w + y * (1 - w) if w != 1 else x
            else:
                x = decoder(x)
        return curr


class ImageGenerator(t.nn.Module):

    def __init__(self, opt):
        super(ImageGenerator, self).__init__()
        ngf = opt.ngf
        norm_layer = get_norm_layer(opt.norm_layer)
        upsampler = get_upsample_layer(opt.upsampler)
        input_dim = opt.input_dim
        output_dim = opt.output_dim
        image_size = opt.image_size
        pref = opt.pref
        padding = opt.padding_type
        n_block = opt.n_block
        ngf_list = [ngf]

        self.n_downsampling = opt.n_downsampling
        self.downsampler = get_downsample_layer()

        # set up encoder
        for i in range(self.n_downsampling):
            if i == 0:
                from_rgb = [Conv(input_dim, pref, kernel_size=7, stride=1, padding=padding), norm_layer(pref),
                            nn.ReLU(True)]
                encoder = [Conv(pref, ngf_list[i], kernel_size=3, stride=2, padding=padding), norm_layer(ngf_list[i]),
                           nn.ReLU(True)]
                setattr(self, 'rgb_from_scale_' + str(i), nn.Sequential(*from_rgb))
                setattr(self, 'encoder_from_scale_' + str(i), nn.Sequential(*encoder))
            else:
                ngf_list += [min(ngf_list[-1] * 2, 512)]
                from_rgb = [Conv(input_dim, ngf_list[-2], kernel_size=7, stride=1, padding=padding),
                            norm_layer(ngf_list[-2]), nn.ReLU(True)]
                encoder = [Conv(ngf_list[-2], ngf_list[-1], kernel_size=3, stride=2, padding=padding),
                           norm_layer(ngf_list[-1]), nn.ReLU(True)]
                setattr(self, 'rgb_from_scale_' + str(i), nn.Sequential(*from_rgb))
                setattr(self, 'encoder_from_scale_' + str(i), nn.Sequential(*encoder))

        # set up res_block
        ngf_list += [ngf_list[-1]]
        res_blocks = []
        for i in range(n_block):
            res_blocks += [ResnetBlock(ngf_list[-1], norm_layer)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # set up decoder
        for i in range(self.n_downsampling, 0, -1):
            if i == 1:
                to_rgb = [Conv(pref, output_dim, kernel_size=1, stride=1, padding=padding), nn.Tanh()]
                decoder = [upsampler(), Conv(ngf_list[i], ngf_list[i - 1], kernel_size=3, stride=1, padding=padding),
                           norm_layer(ngf_list[i - 1]), nn.ReLU(True)]
                setattr(self, 'rgb_to_scale_' + str(i - 1), nn.Sequential(*to_rgb))
                setattr(self, 'decoder_to_scale_' + str(i - 1), nn.Sequential(*decoder))
            else:
                to_rgb = [Conv(ngf_list[i - 1], output_dim, kernel_size=1, stride=1, padding=padding), nn.Tanh()]
                decoder = [upsampler(), Conv(ngf_list[i], ngf_list[i - 1], kernel_size=3, stride=1, padding=padding),
                           norm_layer(ngf_list[i - 1]), nn.ReLU(True)]
                setattr(self, 'rgb_to_scale_' + str(i - 1), nn.Sequential(*to_rgb))
                setattr(self, 'decoder_to_scale_' + str(i - 1), nn.Sequential(*decoder))

    def forward(self, input, scale, w=1):

        if scale == self.n_downsampling - 1:
            assert w == 1

        for i in range(scale, self.n_downsampling):
            encoder = getattr(self, 'encoder_from_scale_' + str(i))
            if i == scale:
                from_rgb = getattr(self, 'rgb_from_scale_' + str(i))
                x = encoder(from_rgb(input))
            elif i == scale + 1 and w != 1:
                from_rgb = getattr(self, 'rgb_from_scale_' + str(i))
                y = from_rgb(self.downsampler(input))
                x = encoder(w * x + (1 - w) * y)
            else:
                x = encoder(x)

        x = self.res_blocks(x)

        for i in range(self.n_downsampling - 1, scale - 1, -1):
            decoder = getattr(self, 'decoder_to_scale_' + str(i))
            if i == scale + 1 and w != 1:
                to_rgb = getattr(self, 'rgb_to_scale_' + str(i))
                next_decoder = getattr(self, 'rgb_to_scale_' + str(i - 1))
                y = next_decoder[0](to_rgb(x))
            elif i == scale:
                to_rgb = getattr(self, 'rgb_to_scale_' + str(i))
                x = to_rgb(decoder(x))
                curr = x * w + y * (1 - w) if w != 1 else x
            else:
                x = decoder(x)
        return curr















