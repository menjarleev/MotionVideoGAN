import os
import torch
from options.train_options import TrainOptions
from models.models import create_model, create_optimizer, init_params, save_models, update_models, init_model_states, update_weights, detach_model_states
from data.data_loader import CreateDataLoader
from subprocess import call
from util.visualizer import Visualizer
from util import util
import time


def train():
    opt = TrainOptions().parse()
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.nThreads = 1

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(dataset)
    print('training videos = %d' % dataset_size)

    models = create_model(opt)
    modelG, modelD, optimizer_G, optimizer_D, optimizer_D_T = create_optimizer(opt, models)

    n_gpus, tG, tD, start_epoch, epoch_iter, print_freq, total_steps, iter_path, input_dim, output_dim = init_params(opt, modelG, modelD, data_loader)
    visualizer = Visualizer(opt)

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        for idx, data in enumerate(dataset, start=epoch_iter):
            if total_steps % print_freq == 0:
                iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size

            save_fake = total_steps % opt.display_freq == 0
            n_frames_total, n_frames_load, t_len = data_loader.dataset.init_data_params(data, n_gpus, tG)
            init_model_states(opt, modelG)
            for i in range(0, n_frames_total, n_frames_load):
                input_A, input_B = data_loader.dataset.prepare_data(data, i, input_dim, output_dim)
                real_A, real_B, fake_B = modelG(input_A, input_B)
                losses = modelD(reshape([real_A, real_B, fake_B]), 'image')
                losses = [torch.mean(x) for x in losses]
                loss_dict = dict(zip(modelD.module.loss_names, losses))
                losses_T = modelD([real_A, real_B, fake_B], 'video')
                losses_T = [torch.mean(x) for x in losses_T]
                loss_dict_T = dict(zip(modelD.module.loss_names_T, losses_T))
                loss_G, loss_D, loss_D_T = modelD.module.get_losses(loss_dict, loss_dict_T)
                backward(opt, loss_G, loss_D, loss_D_T, optimizer_G, optimizer_D, optimizer_D_T)
                detach_model_states(opt, modelG)


            update_weights(opt, total_steps, dataset_size, modelG, modelD)
            if opt.debug:
                call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
            #################### Display results and erros ####################
            ### print errors
            if total_steps % print_freq == 0:
                t = (time.time() - iter_start_time) / print_freq
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                errors.update({k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_T.items()})
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if save_fake:
                visuals = util.save_all_tensors(opt, real_A, fake_B, real_B, modelD)
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD)
            if epoch_iter > dataset_size - opt.batch_size:
                epoch_iter = 0
                break

        visualizer.vis_print('End of epoch %d / %d \t Time Taken: %d sec' %
                             (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch and update model params
        save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD, end_of_epoch=True)
        update_models(opt, epoch, modelG, modelD, data_loader)

def backward(opt, loss_G, loss_D, loss_D_T, optimizer_G, optimizer_D, optimizer_D_T):
    if opt.net_type == 'video' and opt.scale == 0:
        loss_backward(opt, loss_D_T, optimizer_D_T)
    loss_backward(opt, loss_G, optimizer_G)
    loss_backward(opt, loss_D, optimizer_D)


def loss_backward(opt,loss, optimizer):
    optimizer.zero_grad()
    if opt.fp16:
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

def reshape(tensors):
    if tensors is None:
        return None
    if isinstance(tensors, list):
        return [reshape(tensor) for tensor in tensors]
    _, _, ch, h, w = tensors.size()
    return tensors.contiguous().view(-1, ch, h, w)

if __name__ == "__main__":
    train()
