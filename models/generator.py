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
        self.feature_size = opt.image_size
        pref = opt.pref
        padding = opt.padding_type
        n_res_block = opt.n_res_block
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
                encoder = [ConvLSTMBAM(pref), Conv(pref, ngf_list[i], kernel_size=3, stride=2, padding=padding), norm_layer(ngf_list[i]), nn.ReLU(True)]
                setattr(self, 'rgb_from_scale_' + str(i), nn.Sequential(*from_rgb))
                setattr(self, 'encoder_from_scale_' + str(i), nn.Sequential(*encoder))
            else:
                ngf_list += [min(ngf_list[-1] * 2, 512)]
                from_rgb = [Conv(input_dim, ngf_list[-2], kernel_size=7, stride=1, padding=padding), norm_layer(ngf_list[-2]), nn.ReLU(True)]
                encoder = [Conv(ngf_list[-2], ngf_list[-1], kernel_size=3, stride=2, padding=padding), norm_layer(ngf_list[-1]), nn.ReLU(True)]
                setattr(self, 'rgb_from_scale_' + str(i), nn.Sequential(*from_rgb))
                setattr(self, 'encoder_from_scale_' + str(i), nn.Sequential(*encoder))

       # set up res_block
        ngf_list += [ngf_list[-1]]
        res_blocks = []
        for i in range(n_res_block):
            # res_blocks += [ResnetBlock(ngf_list[-1], norm_layer)]
            res_blocks += [RCBAM(ngf_list[-1], ngf_list[-1], norm_layer)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # set up decoder
        for i in range(self.n_downsampling, 0, -1):
            if i == 1:
                to_rgb = [Conv(pref, output_dim, kernel_size=1, stride=1, padding=padding), nn.Tanh()]
                decoder = [nn.Upsample(scale_factor=2, mode=opt.upsample_type, align_corners=False), Conv(ngf_list[i], ngf_list[i-1], kernel_size=3, stride=1, padding=padding), norm_layer(ngf_list[i-1]), nn.ReLU(True),
                           Conv(ngf_list[i-1], pref, kernel_size=3, stride=1, padding=padding), norm_layer(pref), nn.ReLU(True),
                           ConvLSTMBAM(pref)]
                # decoder = [nn.ConvTranspose2d(ngf_list[i], ngf_list[i-1], kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf_list[i-1]), nn.ReLU(True),
                #            Conv(ngf_list[i-1], pref, kernel_size=3, stride=1, padding=padding), norm_layer(pref), nn.ReLU(True),
                #            ConvLSTMBAM(pref, image_size)]
                setattr(self, 'rgb_to_scale_' + str(i - 1), nn.Sequential(*to_rgb))
                setattr(self, 'decoder_to_scale_' + str(i - 1), nn.Sequential(*decoder))
            else:
                to_rgb = [Conv(ngf_list[i-1], output_dim, kernel_size=1, stride=1, padding=padding), nn.Tanh()]
                decoder = [nn.Upsample(scale_factor=2, mode=opt.upsample_type, align_corners=False), Conv(ngf_list[i], ngf_list[i - 1], kernel_size=3, stride=1, padding=padding), norm_layer(ngf_list[i - 1]), nn.ReLU(True)]
                # decoder = [nn.ConvTranspose2d(ngf_list[i], ngf_list[i - 1], kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf_list[i - 1]), nn.ReLU(True)]
                setattr(self, 'rgb_to_scale_' + str(i - 1), nn.Sequential(*to_rgb))
                setattr(self, 'decoder_to_scale_' + str(i - 1), nn.Sequential(*decoder))

    def init_state(self, batch_size, gpu_id):
        self.encoder_state = getattr(self,'encoder_from_scale_0')[0].init_state(batch_size=batch_size, feature_size=self.feature_size, gpu_id=gpu_id)
        self.decoder_state = getattr(self,'decoder_to_scale_0')[-1].init_state(batch_size=batch_size, feature_size=self.feature_size, gpu_id=gpu_id)
        return [self.encoder_state, self.decoder_state]

    def detach_state(self):
        getattr(self, 'encoder_from_scale_0')[0].detach_state()
        getattr(self, 'decoder_to_scale_0')[-1].detach_state()

    def forward(self, input, scale, w=1):
        if scale == self.n_downsampling - 1:
            assert w == 1

        for i in range(scale, self.n_downsampling):
            encoder = getattr(self, 'encoder_from_scale_' + str(i))
            if i == scale:
                from_rgb = getattr(self,'rgb_from_scale_' + str(i))
                x = from_rgb(input)
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
        n_res_block = opt.n_res_block
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
        for i in range(n_res_block):
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






class Encoder(t.nn.Module):

    def __init__(self, opt):
        super(Encoder, self).__init__()
        ngf = opt.ngf
        norm_layer = get_norm_layer(opt.norm_layer)
        input_dim = opt.input_dim
        pref = opt.pref
        padding = opt.padding_type
        ngf_list = [ngf]
        n_res_block = opt.n_res_block // 2

        self.n_downsampling = opt.n_downsampling
        self.downsampler = get_downsample_layer()

        # set up encoder
        for i in range(self.n_downsampling):
            if i == 0:
                from_rgb = [Conv(input_dim, pref, kernel_size=7, stride=1, padding=padding), norm_layer(pref), nn.ReLU(True)]
                encoder = [Conv(pref, ngf_list[i], kernel_size=3, stride=2, padding=padding), norm_layer(ngf_list[i]), nn.ReLU(True)]
                setattr(self, 'rgb_from_scale_' + str(i), nn.Sequential(*from_rgb))
                setattr(self, 'encoder_from_scale_' + str(i), nn.Sequential(*encoder))
            else:
                ngf_list += [min(ngf_list[-1] * 2, 512)]
                from_rgb = [Conv(input_dim, ngf_list[-2], kernel_size=7, stride=1, padding=padding), norm_layer(ngf_list[-2]), nn.ReLU(True)]
                encoder = [Conv(ngf_list[-2], ngf_list[-1], kernel_size=3, stride=2, padding=padding), norm_layer(ngf_list[-1]), nn.ReLU(True)]
                setattr(self, 'rgb_from_scale_' + str(i), nn.Sequential(*from_rgb))
                setattr(self, 'encoder_from_scale_' + str(i), nn.Sequential(*encoder))

        res_blocks = []
        for i in range(n_res_block):
            res_blocks += [RCBAM(ngf_list[-1], ngf_list[-1], norm_layer)]
        self.res_blocks = nn.Sequential(*res_blocks)



    def forward(self, input, scale, w=1):
        if scale == self.n_downsampling - 1:
            assert w == 1

        for i in range(scale, self.n_downsampling):
            encoder = getattr(self, 'encoder_from_scale_' + str(i))
            if i == scale:
                from_rgb = getattr(self,'rgb_from_scale_' + str(i))
                x = from_rgb(input)
                x = encoder(x)
            elif i == scale + 1 and w != 1:
                from_rgb = getattr(self, 'rgb_from_scale_' + str(i))
                y = from_rgb(self.downsampler(input))
                x = encoder(w * x + (1 - w) * y)
            else:
                x = encoder(x)
        x = self.res_blocks(x)
        return x


class RecursiveComponent(t.nn.Module):
    def __init__(self, opt):
        super(RecursiveComponent, self).__init__()
        dim = min(512, opt.ngf * (2 ** (opt.n_downsampling - 1)))
        n_recursive_block = opt.n_recursive_block
        recursive_blocks = []
        padding = opt.padding_type
        for i in range(n_recursive_block):
            recursive_blocks += [RConv(dim, padding=padding)]
        # self.recursive_blocks = nn.ModuleList(recursive_blocks)
        self.recursive_blocks = nn.Sequential(*recursive_blocks)


    def forward(self, x):
        # for i in range(len(self.recursive_blocks)):
        #     x = self.recursive_blocks[i](x)
        # return x
        return self.recursive_blocks(x)

class Decoder(t.nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        norm_layer = get_norm_layer(opt.norm_layer)
        output_dim = opt.output_dim
        padding = opt.padding_type
        pref = opt.pref
        self.n_downsampling = opt.n_downsampling
        ngf = opt.ngf
        ngf_list = [ngf]
        n_res_block = opt.n_res_block
        for i in range(self.n_downsampling - 1):
            ngf_list += [min(ngf_list[-1] * 2, 512)]
        # set up decoder
        for i in range(self.n_downsampling, 0, -1):
            if i == 1:
                to_rgb = [Conv(pref, output_dim, kernel_size=1, stride=1, padding=padding), nn.Tanh()]
                decoder = [nn.Upsample(scale_factor=2, mode=opt.upsample_type, align_corners=False), Conv(ngf_list[i-1], pref, kernel_size=3, stride=1, padding=padding), norm_layer(pref), nn.ReLU(True)]
                setattr(self, 'rgb_to_scale_' + str(i - 1), nn.Sequential(*to_rgb))
                setattr(self, 'decoder_to_scale_' + str(i - 1), nn.Sequential(*decoder))

            else:
                to_rgb = [Conv(ngf_list[i - 2], output_dim, kernel_size=1, stride=1, padding=padding), nn.Tanh()]
                decoder = [nn.Upsample(scale_factor=2, mode=opt.upsample_type, align_corners=False), Conv(ngf_list[i - 1], ngf_list[i - 2], kernel_size=3, stride=1, padding=padding), norm_layer(ngf_list[i - 2]), nn.ReLU(True)]
                setattr(self, 'rgb_to_scale_' + str(i - 1), nn.Sequential(*to_rgb))
                setattr(self, 'decoder_to_scale_' + str(i - 1), nn.Sequential(*decoder))
        res_blocks = []
        for i in range(n_res_block):
            res_blocks += [RCBAM(ngf_list[-1], ngf_list[-1], norm_layer)]
        self.res_blocks = nn.Sequential(*res_blocks)

    def forward(self, input, scale, w=1):
        if scale == self.n_downsampling - 1:
            assert w == 1
        x = self.res_blocks(input)
        for i in range(self.n_downsampling - 1, scale - 1, -1):
            decoder = getattr(self, 'decoder_to_scale_' + str(i))
            if i == scale + 1 and w != 1:
                to_rgb = getattr(self, 'rgb_to_scale_' + str(i))
                next_decoder = getattr(self, 'decoder_to_scale_' + str(i - 1))
                x = decoder(x)
                y = next_decoder[0](to_rgb(x))
            elif i == scale:
                to_rgb = getattr(self, 'rgb_to_scale_' + str(i))
                x = decoder(x)
                x = to_rgb(x)
                x = x * w + y * (1 - w) if w != 1 else x
            else:
                x = decoder(x)
        return x


class VAE(t.nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)

    def forward(self, input, scale=0, w=1):
        return self.decoder(self.encoder(input, scale, w), scale, w)

class RecursiveNet(t.nn.Module):
    def __init__(self, opt):
        super(RecursiveNet, self).__init__()
        self.encoder = Encoder(opt)
        self.recursive = RecursiveComponent(opt)
        self.decoder = Decoder(opt)
        self.status = 'None'
        self.feature_size = [size // (2 ** opt.n_downsampling ) for size in opt.image_size]

    def init_state(self, batch_size, gpu_id):
        self.state = []
        for block in self.recursive.recursive_blocks:
            block.init_state(feature_size=self.feature_size, batch_size=batch_size,  gpu_id=gpu_id)
        return self.state

    def detach_state(self):
        for block in self.recursive.recursive_blocks:
            block.detach_state()

    def train_recursive(self):
        self.encoder.eval()
        self.recursive.train()
        self.decoder.eval()
        if self.status != 'recursive':
            status = 'recursive'
            print('update generator status from %s to %s' % (self.status, status))
            self.status = status

    def train_full(self):
        self.encoder.eval()
        self.recursive.eval()
        self.decoder.eval()
        if self.status != 'full':
            status = 'full'
            print('update generator status from %s to %s' % (self.status, status))
            self.status = status

    def forward(self, input, scale=0, w=1):
        x = self.encoder(input, scale, w)
        x = self.recursive(x)
        x = self.decoder(x, scale, w)
        return x



