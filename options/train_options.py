from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()

    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--pref_video_D', type=int, default=16, help='the output size of first from_RGB block of video_D')
        self.parser.add_argument('--pref_image_D', type=int, default=16, help='the output size of first from_RGB block of image_D')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training result on the console')
        self.parser.add_argument('--print_update_freq', type=int, default=1000, help='frequency of showing how weight of vid dis and upscale change')
        self.parser.add_argument('--phase', type=str, default='train', help='phase of the model, pick from [train, val, test]')
        self.parser.add_argument('--n_frame', type=int, default=3, help='number of frame to feed into g/d per iteration when train video')
        self.parser.add_argument('--n_frames_D', type=int, default=3, help='# of frames feed into video discriminator')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--niter', type=int, default=10, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_recursive', type=int, default=5, help='# of iter to train recursive component')
        self.parser.add_argument('--niter_decay', type=int, default=10, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='momentum term2 of adam')
        self.parser.add_argument('--next_scale', type=bool, default=True, help='is opt.scale next scale or current scale')
        self.parser.add_argument('--continue_train', action='store_true', help='whether continue to train model with current scale')
        self.parser.add_argument('--niter_weight_update', type=int, default=5, help='how many epochs to spend to upscale image gradually')

        # loss term
        self.parser.add_argument('--gan_mode', type=str, default='ls', help='[ls|origin|hinge]')
        self.parser.add_argument('--no_gan_feat', action='store_true', help='do not match discriminator features')
        self.parser.add_argument('--no_vgg', action='store_true', help='do not use perceptual loss')
        self.parser.add_argument('--no_struct', action='store_true', help='do not use structure loss')
        self.parser.add_argument('--no_texture', action='store_true', help='do not use texture loss')
        self.parser.add_argument('--niter_vid_update', type=int, default=3, help='for how many epoch the weight of video discriminator grow up to 1')
        # weight of loss
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight of feature loss for both video/image')
        self.parser.add_argument('--lambda_struct', type=float, default=10.0, help='weight of structure loss for model')
        self.parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        self.parser.add_argument('--lambda_texture', type=float, default=10.0, help='weight of texture loss')


        # temporal
        self.parser.add_argument('--max_frames_per_gpu', type=int, default=3, help='max number of frames per load')
        self.parser.add_argument('--n_frames_G', type=int, default=1, help='# of frames to feed into generator')
        self.parser.add_argument('--n_frames_total', type=int, default=30, help='# of frames per sequence to train with')
        self.parser.add_argument('--max_t_step', type=int, default=1, help='space between neighboring frames')
        self.parser.add_argument('--fp16', action='store_true', help='use MAP precision')
        self.parser.add_argument('--n_frames_bp', type=int, default=3, help='after # of fake images are generated, gradient is cut')

        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')



        self.isTrain = True
