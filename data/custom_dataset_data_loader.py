from .base_data_loader import BaseDataLoader
from torch.utils.data import DataLoader

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'temporal':
        from .temporal_dataset import TemporalDataset
        dataset = TemporalDataset()
    elif opt.dataset_mode == 'pose':
        from .pose_dataset import PoseDataset
        dataset = PoseDataset()
    elif opt.dataset_mode == 'image':
        from .align_dataset import AlignedDataset
        dataset = AlignedDataset()
    else:
        raise ValueError('Dataset [%s] is not recognized' % opt.dataset_mode)
    print('dataset [%s] was created' % dataset.name())
    dataset.initialize(opt)
    return dataset

class CustomDatasetLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads)
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataloader), self.opt.max_dataset_size)
