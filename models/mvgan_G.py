from .base_model import BaseModel
import torch
import torch.nn as nn
from util import util
from .models import define_G

class mvganG(BaseModel):

    def name(self):
        return 'mvganG'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        if not opt.debug:
            # enables benchmark mode (auto-tuner to find best algorithm)
            # the input size should not change at all
            torch.backends.cudnn.benchmark = True

        self.netG = define_G(opt)
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
    def init_states(self, batch_size):
        if self.net_type == 'video':
            self.states = []
            for i in range(len(self.gpus_gen)):
                self.states += [self.netG.init_state(batch_size, self.gpus_gen[i])]

    def detach_states(self):
        if self.net_type == 'video':
            for i in range(len(self.states)):
                self.states[i] = [[s.detach() for s in state] for state in self.states[i]]


    def encode_input(self, input_map, real_image):
        size = input_map.size()
        self.bs, tG, self.input_dim , self.height, self.width = size[0], size[1], size[2], size[3], size[4]
        input_map = input_map.cuda()
        real_image = real_image.cuda()
        return input_map, real_image

    def forward(self, input_A, input_B, dummy_bs=0):
        real_A, real_B= self.encode_input(input_A, input_B)
        if real_A.get_device() == self.gpus_gen[0]:
            real_A, real_B= util.remove_dummy_from_tensor([real_A, real_B], dummy_bs)
            if input_A.size(0) == 0:
                return self.return_dummy(input_A)
        netG = torch.nn.parallel.replicate(self.netG, devices=self.gpus_gen)
        if self.scale == 0:
            for i in range(len(self.gpus_gen)):
                netG[i].encoder_state = self.states[i][0]
                netG[i].decoder_state = self.states[i][1]
        start_gpu = self.gpus_gen[0]
        fake_B = self.generate_frame_train(netG, real_A, real_B, start_gpu)

        return real_A, real_B, fake_B

    def generate_frame_train(self, netG, real_A, real_B, start_gpu):
        tG = self.opt.n_frames_G
        n_frames_per_load = self.n_frames_per_load
        dest_id = self.gpus_out[0] if self.split_gpus else start_gpu
        _, _, bc, bh, bw = real_B.size()
        fake_Bs = None
        for t in range(n_frames_per_load):
            gpu_id = (t // self.n_frames_per_gpu + start_gpu) if self.split_gpus else start_gpu # the GPU idx where we generate this frame
            net_id = t // self.n_frames_per_gpu if self.split_gpus else 0                                           # the GPU idx where the net is located
            _, _, _, h, w = real_A.size()
            input_Ai = real_A[:, t:t+tG, ...].view(self.bs, -1, h, w).cuda(gpu_id)
            fake_B = netG[net_id](input_Ai, self.scale, self.old_w).unsqueeze(1)
            # if self.scale == 0 and (t + 1) % self.n_frames_bp == 0:
            #         netG[0][net_id].detach_state()
            fake_Bs = fake_B if fake_Bs is None else torch.cat([fake_Bs, fake_B.cuda(dest_id)], dim=1)
        return fake_Bs



    def return_dummy(self, input_A):
        h, w = input_A.size()[:3]
        t = self.n_frames_load
        return self.Tensor(1, t, self.opt.input_dim, h, w), self.Tensor(1, t, 3, h, w), self.Tensor(1, t, 3, h, w)



    def downsample_data(self, tensor_list, scale):
        downsample = nn.AvgPool2d(3, stride=2, padding=[1,1], count_include_pad=False)
        tensor_list = [t.view(-1, self.input_dim, self.height, self.width) for t in tensor_list]
        for i in range(scale):
            tensor_list = [downsample(t) for t in tensor_list]
        _, _, self.height, self.width = tensor_list[-1].size()
        return [t.view(self.bs, -1, self.input_dim, self.height, self.width) for t in tensor_list]

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpus_gen)
        self.save_optimizer(self.optimizer_G, 'G', label)





