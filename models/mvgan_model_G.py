from .base_model import BaseModel
import torch
import torch.nn as nn
from util import util
from .models import define_G

class mvganG_vid(BaseModel):

    def name(self):
        return 'mvganG_vid'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        if not opt.debug:
            # enables benchmark mode (auto-tuner to find best algorithm)
            # the input size should not change at all
            torch.backends.cudnn.benchmark = True

        self.netG = define_G(opt, opt.net_type)
        self.net_type = self.opt.net_type
        self.scale = opt.scale
        self.split_gpus = (self.opt.n_gpus_gen < len(self.opt.gpu_ids)) and (self.opt.batch_size == 1)
        self.gpus_gen = self.opt.gpu_ids[:self.opt.n_gpus_gen]
        self.gpus_out = self.opt.gpu_ids[self.opt.n_gpus_gen + 1:] if self.split_gpus else self.gpus_gen



        print('---------- Networks initialized -------------')
        print('---------------------------------------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            self.load_network(self.netG, 'G', opt.which_epoch, opt.load_pretrain)

        if self.isTrain:
            self.old_lr = opt.lr
            self.old_w = 1

            #initialize optimizer G
            params = list(self.netG.parameters())
            beta1, beta2 = opt.beta1, 0.999
            lr = opt.lr
            self.optimizer_G = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))
            if opt.continue_train or opt.load_pretrain:
                self.load_optimizer(self.optimizer_G, 'G', opt.which_epoch, opt.load_pretrain)

        if self.isTrain:
            self.n_gpus = self.opt.n_gpus_gen if self.opt.batch_size == 1 else 1
            self.n_frames_bp = self.opt.n_frames_bp
            self.n_frames_per_gpu = min(self.opt.max_frames_per_gpu, self.opt.n_frames_total // self.n_gpus)
            self.n_frames_per_load = self.n_gpus * self.n_frames_per_gpu
            if self.opt.debug:
                print('training %d frames at once, using %d gpus, frames per gpu = %d' % (self.n_frames_per_load,
                                                                                          self.n_gpus, self.n_frames_per_gpu))

    def encode_input(self, input_map, real_image):
        size = input_map.size()
        self.bs, tG, self.input_dim , self.height, self.width = size[0], size[1], size[2], size[3], size[4]
        input_map = input_map.cuda(self.gpus_gen[0])
        real_image = real_image.cuda(self.gpus_gen[0])
        return input_map, real_image

    def init_state(self, batch_size):
        self.netG.init_state(batch_size, self.gpus_gen[0])

    def detach_state(self):
        self.netG.detach_state()

    def forward(self, input_A, input_B, dummy_bs=0):
        real_A, real_B= self.encode_input(input_A, input_B)
        fake_B = self.generate_frame_train(real_A, real_B)
        return real_A, real_B, fake_B

    def generate_frame_train(self, real_A, real_B):
        tG = self.opt.n_frames_G
        n_frames_per_load = self.n_frames_per_load
        _, _, bc, bh, bw = real_B.size()
        fake_Bs = None
        for t in range(n_frames_per_load):
            _, _, _, h, w = real_A.size()
            input_Ai = real_A[:, t:t+tG, ...].view(self.bs, -1, h, w)
            fake_B = self.netG(input_Ai, self.scale, self.old_w).unsqueeze(1)
            if self.scale == 0 and (t + 1) % self.n_frames_bp == 0:
                self.netG.detach_state()
            fake_Bs = fake_B if fake_Bs is None else torch.cat([fake_Bs, fake_B], dim=1)
        return fake_Bs

    def inference(self, input_A, input_B):
        assert self.scale == 0
        self.netG.eval()
        real_A, real_B = self.encode_input(input_A, input_B)
        fake_B = self.generate_frame_infer(real_A)
        return fake_B, input_A

    def generate_frame_infer(self, real_A):
        fake_Bs = None
        tG = self.opt.n_frames_G
        _, _, _, h, w = real_A.size()
        real_A_reshaped = real_A[0, :tG].view(1, -1, h, w)
        fake_B = self.netG(real_A_reshaped, 0, 1)
        fake_Bs = fake_B if fake_Bs is None else torch.cat([fake_Bs, fake_B], dim=1)
        return fake_Bs

    def return_dummy(self, input_A):
        h, w = input_A.size()[:3]
        t = self.n_frames_load
        return self.Tensor(1, t, self.opt.input_dim, h, w), self.Tensor(1, t, 3, h, w), self.Tensor(1, t, 3, h, w)

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpus_gen)
        self.save_optimizer(self.optimizer_G, 'G', label)

    def train(self):
        pass

class mvganG_img(BaseModel):

    def name(self):
        return 'mvganG_img'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        if not opt.debug:
            # enables benchmark mode (auto-tuner to find best algorithm)
            # the input size should not change at all
            torch.backends.cudnn.benchmark = True
        self.netG = define_G(opt, which_net=opt.net_type)
        self.scale = opt.scale
        self.split_gpus = (self.opt.n_gpus_gen < len(self.opt.gpu_ids)) and (self.opt.batch_size == 1)
        self.gpus_gen = self.opt.gpu_ids[:self.opt.n_gpus_gen]
        self.gpus_out = self.opt.gpu_ids[self.opt.n_gpus_gen + 1:] if self.split_gpus else self.gpus_gen

        print('---------- Networks initialized -------------')
        print('---------------------------------------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            self.load_network(self.netG, 'G', opt.which_epoch, opt.load_pretrain)

        if self.isTrain:
            self.old_lr = opt.lr
            self.old_w = 1

            #initialize optimizer G
            params = list(self.netG.parameters())
            beta1, beta2 = opt.beta1, 0.999
            lr = opt.lr
            self.optimizer_G = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))
            if opt.continue_train or opt.load_pretrain:
                self.load_optimizer(self.optimizer_G, 'G', opt.which_epoch, opt.load_pretrain)

    def encode_input(self, input_map, real_image):
        size = input_map.size()
        self.bs, self.input_dim , self.height, self.width = size[0], size[1], size[2], size[3]
        input_map = input_map.cuda(self.gpus_gen[0])
        real_image = real_image.cuda(self.gpus_gen[0])
        return input_map, real_image

    def forward(self, input_A, input_B, dummy_bs=0):
        real_A, real_B= self.encode_input(input_A, input_B)
        fake_B = self.netG(real_A, scale=self.scale, w=self.old_w)
        return real_A, real_B, fake_B

    def inference(self, input_A, input_B):
        assert self.scale == 0
        self.netG.eval()
        real_A, real_B = self.encode_input(input_A, input_B)
        fake_B = self.netG(real_A)
        return fake_B, input_A

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpus_gen)
        self.save_optimizer(self.optimizer_G, 'G', label)

    def train(self):
        pass
