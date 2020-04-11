import torch.utils.data as data
import numpy as np
from numpy import random
import torchvision.transforms as transforms
from PIL import Image
import torch

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

    def update_training_batch(self, ratio):
        pass

    def init_data_params(self, data, n_gpus, tG):
        opt = self.opt
        _, n_frames_total,  self.height, self.width = data['B'].size()
        n_frames_total = n_frames_total // opt.output_dim
        n_frames_load = opt.max_frames_per_gpu * n_gpus
        n_frames_load = min(n_frames_load, n_frames_total - tG + 1)
        self.t_len = n_frames_load + tG - 1
        return n_frames_total - self.t_len + 1, n_frames_load, self.t_len

    def prepare_data(self, data, i, input_dim, output_dim):
        t_len, height, width = self.t_len, self.height, self.width
        input_A = (data['A'][:, i*input_dim:(i+t_len)*input_dim, ...]).view(-1, t_len, input_dim, height, width)
        input_B = (data['B'][:, i*output_dim:(i+t_len)*output_dim, ...]).view(-1, t_len, input_dim, height, width)
        return [input_A, input_B]

    def init_frame_idx(self, A_paths):
        self.n_of_seqs = min(len(A_paths), self.opt.max_dataset_size)         # number of sequences to train
        self.seq_len_max = max([len(A) for A in A_paths])                     # max number of frames in the training sequences

        self.seq_idx = 0                                                      # index for current sequence
        self.frame_idx = self.opt.start_frame if not self.opt.isTrain else 0  # index for current frame in the sequence
        self.frames_count = []                                                # number of frames in each sequence
        for path in A_paths:
            self.frames_count.append(len(path) - self.opt.n_frames_G + 1)

        self.folder_prob = [count / sum(self.frames_count) for count in self.frames_count]
        self.n_frames_total = self.opt.n_frames_total if self.opt.isTrain else 1
        self.A, self.B, = None, None

    def update_frame_idx(self, A_paths, index):
        if self.opt.isTrain:
            if self.opt.dataset_mode == 'pose':
                seq_idx = np.random.choice(len(A_paths), p=self.folder_prob) # randomly pick sequence to train
                self.frame_idx = index
            else:
                seq_idx = index % self.n_of_seqs
            return None, None, seq_idx
        else:
            self.change_seq = self.frame_idx >= self.frames_count[self.seq_idx]
            if self.change_seq:
                self.seq_idx += 1
                self.frame_idx = 0
                self.A, self.B, = None, None
            return self.A, self.B, self.seq_idx


def get_video_params(opt, n_frames_total, cur_seq_len, index):
        tG = opt.n_frames_G
        if opt.isTrain:
            n_frames_total = min(n_frames_total, cur_seq_len - tG + 1)
            n_gpus = opt.n_gpus_gen if opt.batch_size == 1 else 1
            n_frames_per_load = opt.max_frames_per_gpu * n_gpus
            n_frames_per_load = min(n_frames_total, n_frames_per_load)
            n_loadings = n_frames_total // n_frames_per_load
            n_frames_total = n_frames_per_load * n_loadings + tG - 1
            max_t_step = min(opt.max_t_step , (cur_seq_len - 1) // (n_frames_total - 1))
            t_step = np.random.randint(max_t_step) + 1
            offset_max = max(1, cur_seq_len - (n_frames_total - 1) * t_step)
            if opt.dataset_mode == 'pose':
                start_idx = index % offset_max
            else:
                start_idx = np.random.randint(offset_max)
            if opt.debug:
                print('loading %d frames in total, first frame starting at index %d, space between neighboring frames is %d'
                      % (n_frames_total, start_idx, t_step))
        else:
            n_frames_total = tG
            start_idx = index
            t_step = 1
        return n_frames_total, start_idx, t_step

def make_power_2(n, base=32.0):
    return int(round(n / base) * base)


def get_img_params(opt, size):
    w, h = size
    new_h, new_w = h, w
    if 'resize' in opt.resize_or_crop:
        new_h = new_w = opt.load_size
    elif 'scale_width' in opt.resize_or_crop:
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    elif 'scale_height' in opt.resize_or_crop:
        new_h = opt.load_size
        new_w = opt.load_size * w // h
    elif 'random_scale_width' in opt.resize_or_crop:
        new_w = random.randint(opt.fine_size, opt.load_size + 1)
        new_h = new_w * h // w
    elif 'random_scale_height' in opt.resize_or_crop:
        new_h = random.randint(opt.find_size, opt.load_size + 1)
        new_w = new_h * w // h
    new_w = int(round(new_w / 4)) * 4
    new_h = int(round(new_h / 4)) * 4

    crop_x = crop_y = 0
    crop_w = crop_h = 0
    if 'crop' in opt.resize_or_crop or "scaled_crop" in opt.resize_or_crop:
        if 'crop' in opt.resize_or_crop:
            crop_w = crop_h = opt.find_size
        else:
            if 'width' in opt.resize_or_crop:
                crop_w = opt.fine_size
                crop_h = opt.fine_size * h // w
            else:
                crop_h = opt.fine_size
                crop_w = opt.fine_size * w // h

        crop_w, crop_h = make_power_2(crop_w), make_power_2(crop_h)
        x_span = (new_w - crop_w) // 2
        crop_x = np.maximum(0, np.minimum(x_span*2, int(np.random.randn() * x_span / 3 + x_span)))
        crop_y = random.randint(0, np.minimum(np.maximum(0, new_h - crop_h), new_h // 8))
    else:
        new_w, new_h = make_power_2(new_w), make_power_2(new_h)
    return {'new_size': (new_w, new_h), 'crop_size':(crop_w, crop_h), 'crop_pos': (crop_x, crop_y)}

def __scale_image(img, size, method=Image.BICUBIC):
    w, h = size
    return img.resize((w, h), method)
def __crop(img, size, pos):
    ow, oh = img.size
    tw, th = size
    x1, y1 = pos
    if(ow > tw or oh > th):
        return img.crop((x1, y1, min(ow, x1 + tw), min(oh, y1 + th)))
    return img

def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    if opt.scale != 0:
        osize = [opt.image_size[0] // pow(2, opt.scale), opt.image_size[1] // pow(2, opt.scale)]
        transform_list.append(transforms.Resize(osize, method))
    elif 'resize' in opt.resize_or_crop:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    else:
        transform_list.append(transforms.Lambda(lambda img: __scale_image(img, params['new_size'], method)))

    if 'crop' in opt.resize_or_crop or 'scaled_crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_size'], params['crop_pos'])))

    if toTensor:
        transform_list +=[transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

def concat_frame(A, Ai, nF):
    if A is None:
        A = Ai
    else:
        c = Ai.size()[0]
        if A.size()[0] == nF * c:
            A = A[c:]
        A = torch.cat([A, Ai])
    return A