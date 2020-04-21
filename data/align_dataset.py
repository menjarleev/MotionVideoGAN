import os.path
from .base_dataset import BaseDataset, get_transform, get_img_params
from .image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.A_is_label = self.opt.label_nc != 0

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))
        assert len(self.A_paths) == len(self.B_paths)

        # ### instance maps
        # if not opt.no_instance:
        #     self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
        #     self.inst_paths = sorted(make_dataset(self.dir_inst))
        #
        # ### load precomputed instance-wise encoded features
        # if opt.load_features:
        #     self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
        #     print('----------- loading features from %s ----------' % self.dir_feat)
        #     self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        ### input A (label maps)
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        B_img = Image.open(B_path).convert('RGB')
        params = get_img_params(self.opt, B_img.size)
        transform_scaleB = get_transform(self.opt, params)
        transform_scaleA = get_transform(self.opt, params, method=Image.NEAREST, normalize=False) if self.A_is_label else transform_scaleB
        A = self.get_image(A_path, transform_scaleA)
        B = self.get_image(B_path, transform_scaleB)

        return_list = {'A': A, 'B': B, 'A_path': A_path, 'B_path': B_path}
        return return_list

    def __len__(self):
        return len(self.A_paths) // self.opt.batch_size * self.opt.batch_size

    def get_image(self, A_path, transform_scaleA, is_label=False):
        A_img = Image.open(A_path)
        A_scaled = transform_scaleA(A_img)
        if is_label:
            A_scaled *= 255.0
        return A_scaled

    def name(self):
        return 'AlignedDataset'