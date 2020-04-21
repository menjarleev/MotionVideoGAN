
import os
from collections import OrderedDict
from options.test_options import TestOptions
from torch.autograd import Variable
from data.data_loader import CreateDataLoader
from models.models import create_model, init_model_state
import util.util as util
from util.visualizer import Visualizer

def test():
    opt = TestOptions.parse(save=False)
    opt.nThreads = 1
    opt.batch_size = 1
    opt.serial_batches = True
    input_dim = opt.input_dim
    output_dim = opt.output_dim
    if opt.dataset_mode == 'temporal':
        opt.dataset_mode = 'test'

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)

    save_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    print('Doing %d frames' % len(dataset))
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        _, _, height, width = data['A'].size()
        A = Variable(data['A'].view(1, -1, input_dim, height, width))
        B = Variable(data['B'].view(1, -1, output_dim, height, width)) if len(data['B'].size()) > 2 else None
        init_model_state(opt, model)
        generated = model.inference(A, B)
        real_A = util.tensor2im(generated[1], normalize=False)
        visual_list = [('real_A', real_A),
                       ('fake_B', util.tensor2im(generated[0].data[0]))]
        visuals = OrderedDict(visual_list)
        img_path = data['A_path']
        print('process image... %s' % img_path)
        visualizer.save_images(save_dir, visuals, img_path)


