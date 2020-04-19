from .base_model import BaseModel
from .models import define_D
from .model_helper import GANLoss, VGGLoss, StructLoss, TextureLoss
from util import util
import torch

class mvganD(BaseModel):
    def name(self):
        return 'mvgan_D'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if not opt.debug:
            torch.backends.cudnn.benchmark = True
        self.tD = opt.n_frames_D
        self.output_dim = opt.output_dim
        self.input_dim = opt.input_dim
        self.netD_img = define_D(opt, 'image')
        self.net_type = opt.net_type
        self.scale = opt.scale
        self.old_w = 1.0
        self.split_gpus = (self.opt.n_gpus_gen < len(self.opt.gpu_ids)) and (self.opt.batch_size == 1)
        self.gpus_dis = self.opt.gpu_ids[self.opt.n_gpus_gen + 1:] if self.split_gpus else self.gpu_ids
        if opt.net_type == 'video':
            self.netD_vid = define_D(opt, 'video')
        elif opt.net_type =='image':
            self.tD = 1

        print('---------- Networks initialized -------------')
        print('-----------------------------------------------')

        beta1, beta2 = opt.beta1, opt.beta2
        self.old_lr = opt.lr
        self.optimizer_D = torch.optim.Adam(self.netD_img.parameters(), lr=self.old_lr, betas=(beta1, beta2))
        self.optimizer_D_T = torch.optim.Adam(self.netD_vid.parameters(), lr=self.old_lr, betas=(beta1, beta2)) if opt.net_type == 'video' else None

        if opt.continue_train or opt.load_pretrain:
            self.load_network(self.netD_img, 'D', opt.which_epoch, opt.load_pretrain)
            self.load_optimizer(self.optimizer_D, 'D', opt.which_epoch, opt.load_pretrain)
            if opt.net_type == 'video':
                self.load_network(self.netD_vid, 'D_T', opt.which_epoch, opt.load_pretrain)
                self.load_optimizer(self.optimizer_D_T, 'D_T', opt.which_epoch, opt.load_pretrain)


        # define loss terms
        self.criterionGAN = GANLoss(opt.gan_mode, tensor=self.Tensor)
        self.criterionStruct = StructLoss()
        self.criterionTexture = TextureLoss()
        self.criterionFeat = torch.nn.L1Loss()

        if not opt.no_vgg:
            self.criterionVGG = VGGLoss(self.gpus_dis[0])

        self.loss_names = ['G_VGG', 'G_GAN', 'G_GAN_Feat', 'G_Struct', 'G_Texture',
                           'D_real', 'D_fake']
        if opt.net_type:
            self.loss_names_T = ['D_T_real', 'D_T_fake', 'G_T_GAN', 'G_T_GAN_Feat']



    def forward(self, tensor_list, type='video', dummy_bs=0):
        lambda_vgg = self.opt.lambda_vgg
        lambda_struct = self.opt.lambda_struct
        lambda_texture = self.opt.lambda_texture
        real_A, real_B, fake_B = tensor_list
        if tensor_list[0].get_device() == self.gpus_dis[0]:
            tensor_list = util.remove_dummy_from_tensor(tensor_list, dummy_bs)
            if tensor_list[0].size(0) == 0:
                return [self.Tensor(1, 1).fill_(0.0) * (len(self.loss_name_T) if type=='video' else len(self.loss_name))]
        if type == 'video':
            if self.net_type == 'video' and self.scale == 0:
                loss_D_T_real, loss_D_T_fake, loss_G_T_GAN, loss_G_T_Feat = self.compute_loss_D_T(self.netD_vid, real_B, fake_B, real_A)
                loss_list = [loss_D_T_real, loss_D_T_fake, loss_G_T_GAN, loss_G_T_Feat ]
                loss_list = [loss.view(-1, 1) for loss in loss_list]
                return loss_list
            else:
                return [self.Tensor(1, 1).fill_(0.0)] * len(self.loss_names_T)


        _, _, self.height, self.width = real_B.size()
        loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat = self.compute_loss_D(self.netD_img, real_A, real_B, fake_B)
        loss_G_VGG = (self.criterionVGG(fake_B, real_B) * lambda_vgg) if not self.opt.no_vgg else self.Tensor(1,1).fill_(0.0)
        loss_G_Struct = (self.criterionStruct(fake_B, real_B) * lambda_struct) if not self.opt.no_struct else self.Tensor(1,1).fill_(0.0)
        loss_G_Texture = (self.criterionTexture(fake_B, real_B, real_A) * lambda_texture) if not self.opt.no_texture else self.Tensor(1,1).fill_(0.0)
        loss_list = [loss_G_VGG, loss_G_GAN, loss_G_GAN_Feat, loss_G_Struct, loss_G_Texture, loss_D_real, loss_D_fake]
        loss_list = [loss.view(-1, 1) for loss in loss_list]
        return loss_list


    def save(self, label):
        self.save_network(self.netD_img, 'D', label, self.gpus_dis)
        self.save_optimizer(self.optimizer_D, 'D', label)
        if self.net_type == 'video':
            self.save_network(self.netD_vid, 'D_T', label, self.gpus_dis)
            self.save_optimizer(self.optimizer_D_T, 'D_T', label)

    def get_losses(self, loss_dict, loss_dict_T):
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG'] \
                 + loss_dict['G_Struct'] + loss_dict['G_Texture'] + loss_dict_T['G_T_GAN'] \
                 + loss_dict_T['G_T_GAN_Feat']
        loss_D_T = (loss_dict_T['D_T_real'] + loss_dict_T['D_T_fake']) * 0.5
        return loss_G, loss_D, loss_D_T

    def GAN_and_FM_loss(self, pred_real, pred_fake):
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        loss_G_GAN_Feat = torch.zeros_like(loss_G_GAN)
        if not self.opt.no_gan_feat:
            feat_weights = 4.0 / (self.opt.n_downsampling + 1)
            for i in range(len(pred_fake) - 1):
                loss_G_GAN_Feat += feat_weights * self.criterionFeat(pred_fake[i], pred_real[i].detach()) \
                                   * self.opt.lambda_feat
        return loss_G_GAN, loss_G_GAN_Feat


    def compute_loss_D(self, netD, real_A, real_B, fake_B):
        real_AB = torch.cat((real_A, real_B), dim=1)
        fake_AB = torch.cat((real_A, fake_B), dim=1)
        pred_real = netD(real_AB, self.scale, self.old_w)
        pred_fake = netD(fake_AB.detach(), self.scale, self.old_w)
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        pred_fake = netD(fake_AB, self.scale, self.old_w)
        loss_G_GAN, loss_G_GAN_Feat = self.GAN_and_FM_loss(pred_real, pred_fake)
        return loss_D_real, loss_D_fake, loss_G_GAN, loss_G_GAN_Feat

    def compute_loss_D_T(self, netD_T, real_B, fake_B, real_A):
        real_B = real_B.view(-1, self.output_dim * self.tD, self.height, self.width)
        fake_B = fake_B.view(-1, self.output_dim * self.tD, self.height, self.width)
        real_A = real_A.view(-1, self.output_dim * self.tD, self.height, self.width)
        concat_real = torch.cat([real_B, real_A], dim=1)
        concat_fake = torch.cat([fake_B, real_A], dim=1)
        pred_real = netD_T(concat_real, self.scale, self.old_w)
        pred_fake = netD_T(concat_fake.detach(), self.scale, self.old_w)
        loss_D_T_real = self.criterionGAN(pred_real, True)
        loss_D_T_fake = self.criterionGAN(pred_fake, False)
        pred_fake = netD_T(concat_fake, self.scale, self.old_w)
        loss_G_T_GAN, loss_G_T_GAN_Feat = self.GAN_and_FM_loss(pred_real, pred_fake)
        return loss_D_T_real, loss_D_T_fake, loss_G_T_GAN, loss_G_T_GAN_Feat

