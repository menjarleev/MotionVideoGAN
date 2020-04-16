import torch as t
import os

class BaseModel(t.nn.Module):
    """
    base module for all network, encapsulate save & load method
    """
    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids
        self.Tensor = t.cuda.FloatTensor if self.gpu_ids else t.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def get_curr_visuals(self):
        return self.input

    def get_curr_errors(self):
        return {}

    def save(self, label):
        pass

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        file_name = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, file_name)
        t.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and t.cuda.is_available():
            network.cuda(gpu_ids[0])

    def save_optimizer(self, optimizer, network_label, epoch_label):
        file_name = '%s_optimizer_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, file_name)
        t.save(optimizer.state_dict(), save_path)

    def load_optimizer(self, optimizer, network_label, epoch_label, save_dir=''):
        file_name = '%s_optimizer_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, file_name)
        if not os.path.join(save_path):
            print('%s does not exist!' % save_path)
        else:
            try:
                optimizer.load_state_dict(t.load(save_path))
            except:
                raise ValueError('optimzer parameters does not fit!')

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        file_name = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, file_name)
        if not os.path.isfile(save_path):
            print('%s does not exist!' % save_path)
        else:
            try:
                network.load_state_dict(t.load(save_path))
            except:
                pretrained_dict = t.load(save_path)
                model_dict = network.state_dict()

                initilized = set()
                for k, v in pretrained_dict.items():
                    initilized.add(k.split('.')[0])
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    print('Pretrained network %s has excessive layers. Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers. The following are not initialized: ' % network_label)
                    not_initialized = set()
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    def update_learning_rate(self, epoch, model):
        lr = self.opt.lr * (1 - (epoch - self.opt.niter) / self.opt.niter_decay)
        for param_group in getattr(self, 'optimizer_' + model).param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_weight(self, step_decay_ratio, epoch_iter, step_length):
        if self.scale == 0 or self.scale == self.opt.n_downsampling - 1:
            self.old_w = 1
            return
        elif epoch_iter <= step_decay_ratio * step_length:
            w = epoch_iter / (step_decay_ratio * step_length)
        else:
            self.old_w = 1
            return
        print('update model weight: %f -> %f' % (self.old_w, w))
        self.old_w = w

    def update_scale(self, scale):
        print('update scale: %f -> %f' % (self.scale, scale))
        self.scale = scale


    def concat(self,  tensors, dim=0):
        if tensors[0] is not None and tensors[1] is not None:
            tensors_cat =[]
            for i in range(len(tensors[0])):
                tensors_cat.append(self.concat([tensors[0][i], tensors[1][i]], dim=dim))
            return tensors_cat
        elif tensors[0] is not None:
            return tensors[0]
        else:
            return tensors[1]
