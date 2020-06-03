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
        self.Tensor = t.cuda.FloatTensor if self.gpu_ids else t.FloatTensor
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

    def load_optimizer(self, optimizer, optimizer_label, epoch_label, save_dir=''):
        file_name = '%s_optimizer_%s.pth' % (epoch_label, optimizer_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, file_name)
        if not os.path.join(save_path):
            print('%s does not exist!' % save_path)
        else:
            try:
                optimizer.load_state_dict(t.load(save_path))
            except:
                print('Optimizer %s parameters does not match, ignore loading optimizer' % optimizer_label)
                # pretrained_dict = t.load(save_path)
                # model_dict = optimizer.state_dict()
                # initialized = set()
                # for k, v in pretrained_dict.items():
                #     initialized.add(k.split('.')[0])
                # try:
                #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                #     optimizer.load_state_dict(pretrained_dict)
                #     print('Optimizer %s has excessive parameters. Only loading parameters that are used' % optimizer_label)
                # except:
                #     print('Optimizer %s has fewer layers. The following are not initialized: ' % optimizer_label)
                #     not_initialized = set()
                #     for k, v in pretrained_dict.items():
                #         if v.size() == model_dict[k].size():
                #             model_dict[k] = v
                #
                #     for k, v in model_dict.items():
                #         if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                #             not_initialized.add(k.split('.')[0])
                #     print(sorted(not_initialized))
                #     optimizer.load_state_dict(model_dict)

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

                initialized = set()
                for k, v in pretrained_dict.items():
                    initialized.add(k.split('.')[0])
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

    def update_weight(self, total_steps, step_length):
        if not self.isTrain or self.scale == self.opt.n_scale - 1:
            self.old_w = 1
            return self.old_w, self.old_w
        elif total_steps <= self.opt.niter_weight_update * step_length:
            w = total_steps / (self.opt.niter_weight_update * step_length)
            old_w = self.old_w
            new_w = w
            self.old_w = w
            return old_w, new_w
        else:
            self.old_w = 1
            return self.old_w, self.old_w

    def update_phi(self, total_steps, step_length):
        if not self.isTrain:
            self.old_phi = 1
            return self.old_phi, self.old_phi
        elif total_steps <= self.opt.niter_phi_update * step_length:
            phi = total_steps / (self.opt.niter_phi_update * step_length)
            old_phi = self.old_phi
            new_phi = phi
            self.old_phi = phi
            return old_phi, new_phi
        else:
            self.old_phi = 1
            return self.old_phi, self.old_phi

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

