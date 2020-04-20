from .base_options import BaseOptions
class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        self.parser.add_argument('--results_dir', type='str', default='./results/', help='saves results here.')
        self.parser.add_argument('--start_frame', type=int, default=0, help='frame index to start inference on')
        self.parser.add_argument('--how_many', type=int, default=float('inf'), help='how many test images to run')
        self.isTrain = False