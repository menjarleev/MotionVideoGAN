import argparse
import os
import torch as t

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, default='./', help='dataroot of model')
        # hyper-parameter
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--load_size', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--fine_size', type=int, default=512, help='lower bound of resolution for input image')
        self.parser.add_argument('--resize_or_crop', type=str, default='', help='crop or resize dataset')
        self.parser.add_argument('--norm_G', type=str, default='spectralinstance', help='instance normalization or batch normalization')

        # model structure
        self.parser.add_argument('--n_scale', type=int, default=3, help='number of encoder & decoder in the model')
        self.parser.add_argument('--n_downsampling', type=int, default=4, help='number of downsample after n_scale in the model')
        self.parser.add_argument('--padding_type', type=str, default='reflect', help='padding type, select from [reflect, replicate, zero]')
        self.parser.add_argument('--norm_layer', type=str, default='instance', help='the norm layer of model, choose from [batch, instance, None]')
        self.parser.add_argument('--upsampler', type=str, default='bicubic', help='the upsampler for decoder, choose from [nearest, linear, bilinear, bicubic]')

        # generator
        self.parser.add_argument('--pref', type=int, default=8, help='number of out channel before encoder & after decoder')
        self.parser.add_argument('--ngf', type=int, default=32, help='number of initial generator kernel')
        self.parser.add_argument('--n_res_block', type=int, default=8, help='number of ResBlock in the network')
        self.parser.add_argument('--n_mask_block', type=int, default=4, help='number of ResBlock in mask branch')
        self.parser.add_argument('--n_recursive_block', type=int, default=4, help='number of Rconv in the network')
        self.parser.add_argument('--stage_one_block', type=int, default=9, help='number of ResBlock in the network')
        self.parser.add_argument('--stage_two_block', type=int, default=3, help='number of ResBlock in the network')

        # discriminator
        self.parser.add_argument('--ndf', type=int, default=64, help='number of initial discriminator kernel')
        self.parser.add_argument('--negative_slope', type=float, default=0.2, help='the negative slope value for LeakyReLU')

        # input & output info
        self.parser.add_argument('--image_size', type=str, default='512, 1024', help='the size of image, if W != H, it will be cropped')
        self.parser.add_argument('--input_dim', type=int, default=3, help='number of input channel')
        self.parser.add_argument('--output_dim', type=int, default=3, help='number of output channel')
        self.parser.add_argument('--upsample_type', type=str, default='bicubic', help='which upsample method to use')
        self.parser.add_argument('--max_channel', type=int, default=256, help='max # of channel for each layer')

        self.parser.add_argument('--densepose_only', action='store_true', help='only use dense pose')
        self.parser.add_argument('--openpose_only', action='store_true', help='only use open pose')
        self.parser.add_argument('--start_frame', type=int, default=0, help='start frame of each sequence in pose dataset')

        self.parser.add_argument('--max_dataset_size', type=int, default=float('inf'), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only subset is loaded')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches')
        self.parser.add_argument('--dataset_mode', type=str, default='temporal', help='chooses how datasets are loaded.')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='if specified, load the pretrained model')
        self.parser.add_argument('--debug', action='store_true', help='activate debug, change display freq')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='where to save checkpoints')
        self.parser.add_argument('--name', type=str, default='experiment', help='the experiment name')
        self.parser.add_argument('--model', type=str, default='mvgan_img', help='the name of the model choose from [mvgan_img, mvgan_vid]')
        self.parser.add_argument('--net_type', type=str, default='branch', help='type of network, choose from [VAE, branch, recursive, image, video, stage1, stage2]')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0,1,2 use -1 for CPU')
        self.parser.add_argument('--n_gpus_gen', type=int, default=1, help='# of gpu used for generator')
        self.parser.add_argument('--scale', type=int, default=0, help='the scale of network to use')
        self.parser.add_argument('--add_face_disc', action='store_true', help='add face region GAN')
        self.parser.add_argument('--label_nc', type=int, default=0, help='number of label channel, default to 0 to not use it')
        self.parser.add_argument('--blur_ratio', type=int, default=0, help='ratio to downscale and upscale the image')


        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--random_drop_prob', type=float, default=0.05, help='the probability to randomly drop each pose segment during training')


        self.initialized = True



    def parse_str(self, ids):
        str_ids = ids.split(',')
        ids_list = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                ids_list.append(id)
        return ids_list

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain
        self.opt.gpu_ids = self.parse_str(self.opt.gpu_ids)
        self.opt.image_size = self.parse_str(self.opt.image_size)
        if self.opt.n_gpus_gen == -1:
            self.opt.n_gpus_gen = len(self.opt.gpu_ids)

        if len(self.opt.gpu_ids) > 0:
            t.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)
        print('--------------- Options ---------------')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print('----------------- End -----------------')

        # make dir to save all the info about this model
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        if self.opt.debug:
            if os.path.isdir(expr_dir):
                print("%s exists" % expr_dir)
            else:
                os.makedirs(expr_dir)
        elif self.opt.continue_train:
            if os.path.isdir(expr_dir):
                print('continue train with dir %s' % expr_dir)
        else:
            if os.path.isdir(expr_dir):
                raise FileNotFoundError('%s exists' %expr_dir)
            else:
                os.makedirs(expr_dir)

        # save opt
        if save:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'w+') as opt_file:
                opt_file.write('--------------- Options ---------------')
                for k, v in args.items():
                    opt_file.write('%s: %s' % (str(k), str(v)))
                opt_file.write('----------------- End -----------------')
        return self.opt






