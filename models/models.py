import torch.nn as nn
import torch
import numpy as np
from .generator import ImageGenerator, VideoGenerator, VAE, RecursiveNet, GlobalGenerator, BranchGenerator, StageOneGenerator, StageTwoGenerator
from .discriminator import VideoDiscriminator, ImageDiscriminator, MultiscaleDiscriminator, WeightedMultiscaleDiscriminator
from .model_helper import weights_init
from .generator import vid_net, img_net

import os
import fractions
def lcm(a,b): return (abs(a*b) / fractions.gcd(a,b)) if a and b else 0

def wrap_model(opt, modelG, modelD):
    # TODO wrap generator in a GPU and discriminator, other pretrained model in the other GPU to avoid CUDA out of memory
    if opt.n_gpus_gen == len(opt.gpu_ids):
        modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids)
        modelD = nn.DataParallel(modelD, device_ids=opt.gpu_ids)
    else:
        if opt.net_type in vid_net:
            gpu_split_id = opt.n_gpus_gen + 1
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[0:1], output_device=opt.gpu_ids[1])
        else:
            gpu_split_id = opt.n_gpus_gen
            modelG = nn.DataParallel(modelG, device_ids=opt.gpu_ids[:gpu_split_id], output_device=opt.gpu_ids[gpu_split_id])
        modelD = nn.DataParallel(modelD, device_ids=opt.gpu_ids[gpu_split_id:])
    return modelG, modelD

class myModel(nn.Module):
    def __init__(self, opt, model):
        super(myModel, self).__init__()
        self.opt = opt
        self.module = model
        self.model = nn.DataParallel(model, device_ids=opt.gpu_ids)
        self.bs_per_gpu = int(np.ceil(float(opt.batch_size)) / len(opt.gpu_ids))
        self.pad_bs = self.bs_per_gpu * len(opt.gpu_ids) - opt.batch_size

    def forward(self, *inputs, **kwargs):
        inputs = self.add_dummy_to_tensor(inputs, self.pad_bs)
        outputs = self.model(*inputs, **kwargs, dummy_bs=self.pad_bs)
        if self.pad_bs == self.bs_per_gpu:
            return self.remove_dummy_from_tensor(outputs, 1)
        return outputs



    def add_dummy_to_tensor(self, tensors, add_size=0):
        if add_size == 0 or tensors is None:
            return tensors
        if type(tensors) == list or type(tensors) == tuple:
            return [self.add_dummy_to_tensor(tensor, add_size) for tensor in tensors]

        if isinstance(tensors, torch.Tensor):
            dummy = torch.zeros_like(tensors)[:add_size]
            tensors = torch.cat([dummy, tensors])
        return tensors

    def remove_dummy_from_tensor(self, tensors, remove_size=0):
        if remove_size == 0 or tensors is None:
            return tensors
        if type(tensors) == list or type(tensors) == tuple:
            return [self.remove_dummy_from_tensor(tensor, remove_size) for tensor in tensors]

        if isinstance(tensors, torch.Tensor):
            tensors = tensors[remove_size:]
        return tensors

def create_model(opt):
    print(opt.model)
    if opt.model == 'mvgan_vid':
        from .mvgan_model_G import mvganG_vid
        modelG = mvganG_vid()
    elif opt.model == 'mvgan_img':
        from .mvgan_model_G import mvganG_img
        modelG = mvganG_img()
    if opt.isTrain:
        from .mvgan_model_D import mvganD
        modelD = mvganD()
    else:
        raise ValueError("Model [%s] not recognized" % opt.model)

    modelG.initialize(opt)
    if opt.isTrain:
        modelD.initialize(opt)
        # todo warp_model
        if not opt.fp16:
            modelG, modelD = wrap_model(opt, modelG, modelD)
        return [modelG, modelD]
    else:
        return modelG

def create_optimizer(opt, models):
    modelG, modelD = models
    optimizer_D_T = None
    if opt.fp16:
        from apex import amp
        modelG, optimizer_G = amp.initialize(modelG, modelG.optimizer_G, opt_level='O1')
        if opt.net_type in vid_net:
            modelD, optimizers_D = amp.initialize(modelD, [modelD.optimizer_D, modelD.optimizer_D_T], opt_level='O1')
            optimizer_D, optimizer_D_T = optimizers_D
        if opt.net_type in img_net:
            modelD, optimizer_D = amp.initialize(modelD, modelD.optimizer_D, opt_level='O1')
        modelG, modelD = wrap_model(opt, modelG, modelD)
    else:
        optimizer_G = modelG.module.optimizer_G
        optimizer_D = modelD.module.optimizer_D
        if opt.net_type in vid_net:
            optimizer_D_T = modelD.module.optimizer_D_T
    return modelG, modelD, optimizer_G, optimizer_D, optimizer_D_T



def define_G(opt, which_net):
    netG = None
    if which_net == 'image':
        netG = ImageGenerator(opt)
    elif which_net == 'video':
        netG = VideoGenerator(opt)
    elif which_net == 'VAE':
        netG = VAE(opt)
    elif which_net == 'recursive':
        netG = RecursiveNet(opt)
    elif which_net == 'pix2pixHD':
        netG = GlobalGenerator(opt)
    elif which_net == 'branch':
        netG = BranchGenerator(opt)
    elif which_net == 'stage1':
        netG = StageOneGenerator(opt)
    elif which_net == 'stage2':
        netG = StageTwoGenerator(opt)
    else:
        raise NotImplementedError('Generator named [%s] is not implemented' % opt.net_type)
    if len(opt.gpu_ids) > 0:
        netG.cuda(opt.gpu_ids[0])
    netG.apply(weights_init)
    if opt.debug:
        print(netG)
    return netG

def define_D(opt, which_model_netD):
    if which_model_netD == 'video':
        netD = VideoDiscriminator(opt)
    elif which_model_netD == 'image':
        netD = ImageDiscriminator(opt)
    elif which_model_netD == 'multiscale':
        netD = MultiscaleDiscriminator(opt, 1)
    elif which_model_netD == 'multiscale_vid':
        netD = MultiscaleDiscriminator(opt, opt.n_frames_D)
    elif which_model_netD == 'weighted_multiscale':
        netD = WeightedMultiscaleDiscriminator(opt)
    else:
        raise NotImplementedError('Discriminator named [%s] is not implemented' % which_model_netD)
    if len(opt.gpu_ids) > 0:
        netD.cuda(opt.gpu_ids[0])
    netD.apply(weights_init)
    if opt.debug:
        print(netD)
    return netD






def init_params(opt, modelG, modelD, data_loader):
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    start_epoch, epoch_iter = 1, 0
    step_per_epoch = len(data_loader)
    if opt.continue_train:
        if os.path.exists(iter_path):
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        if start_epoch > opt.niter:
            modelG.module.update_learning_rate(start_epoch-1, 'G')
            modelD.module.update_learning_rate(start_epoch-1, 'D')
        if modelG.module.scale != opt.scale:
            modelG.module.update_scale(opt.scale)
            modelD.module.update_scale(opt.scale)

    n_gpus = opt.n_gpus_gen if opt.batch_size == 1 else 1
    tG, tD = opt.n_frames_G, opt.n_frames_D
    input_dim = opt.input_dim
    output_dim = opt.output_dim
    print_freq = lcm(opt.print_freq, opt.batch_size)
    total_steps = (start_epoch - 1) * len(data_loader) + epoch_iter
    total_steps = total_steps // print_freq * print_freq
    update_weights(opt, total_steps, len(data_loader), modelG, modelD)
    return n_gpus, tG, tD, start_epoch, epoch_iter, print_freq, total_steps, iter_path, input_dim, output_dim

def update_models(opt, epoch, modelG, modelD, data_loader):
    if epoch > opt.niter:
        modelG.module.update_learning_rate(epoch, 'G')
        modelD.module.update_learning_rate(epoch, 'D')


def update_model_weights(opt, total_step, step_length, modelG, modelD):
    old, new = modelG.module.update_weight(total_step, step_length)
    _, _ = modelD.module.update_weight(total_step, step_length)
    if old != new and (total_step % opt.print_update_freq == 0 or opt.debug):
        print('update upscale model weight from %.4f to %.4f' %(old, new))

def update_d_vid_weights(opt, total_step, step_length, modelD):
    old, new = modelD.module.update_vid_weights(total_step, step_length)
    if old != new and (total_step % opt.print_update_freq == 0 or opt.debug):
        print('update video discriminator weight from %.4f to %.4f' %(old, new))

def has_weight(opt):
    if opt.net_type == 'video' or opt.net_type == 'image' or opt.net_type == 'VAE' or opt.net_type == 'stage2':
        return True
    else:
        return False

def update_weights(opt, total_step, dataset_size, modelG, modelD):
    if has_weight(opt):
        update_model_weights(opt, total_step, dataset_size, modelG, modelD)
        if opt.net_type == 'video':
            update_d_vid_weights(opt, total_step, dataset_size, modelD)


def save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD, end_of_epoch=False):
    if not end_of_epoch:
        if total_steps % opt.save_latest_freq == 0:
            modelG.module.save('latest')
            modelD.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
        else:
            if epoch % opt.save_epoch_freq == 0:
                modelG.module.save('latest')
                modelD.module.save('latest')
                # modelG.module.save(epoch)
                # modelD.module.save(epoch)
                np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

def have_state(opt):
    if (opt.net_type == 'video' or opt.net_type == 'recursive' and opt.scale == 0) or opt.net_type == 'branch' or opt.net_type == 'stage2':
        return True
    else:
        return False

def init_model_state(opt, model):
    if have_state(opt):
        model.module.init_state(opt.batch_size)

def detach_model_state(opt, model):
    if have_state(opt):
        model.module.detach_state()
