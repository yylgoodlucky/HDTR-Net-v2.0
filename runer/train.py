import argparse
import torch, os
import numpy as np

from os.path import join
from tqdm import tqdm
from collections import OrderedDict
from model.configuration import Model_config
from utils.train_util import *
from utils.data_util import *

class Train_pipeline(Model_config):
    def __init__(self, opt, mode):
        super().__init__(opt, mode)
        
        # setup
        self.build_net()
        self.build_data()
        self.build_loss()
        self.build_optimizers()
        
        self.resume_state()
        
    def save_img(self):
        """ Visualization during training """
        # transfr b, c, h, w : tensor -> b, h, w, c : numpy
        save_input, save_ouput = tensor2img([self.input, self.output], rgb2bgr=True, min_max=(-1, 1))
        save_input, save_ouput = eliminate_batch([save_input, save_ouput])

        save_path = join(self.vis_path, str(current_epoch) + '_' + str(current_iter) + '.png')
        cv2.imwrite(save_path, np.concatenate((save_input, save_ouput), axis=1))

    def save_ckpt(self, net, label, param_key='params'):
        """Save networks.
        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """

        save_filename = f'{self.mode}_latest_{label}.pth'
        save_path = os.path.join(self.model_path, save_filename)
        # print(f'[info] : save ckpt in {save_path}')

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        torch.save(save_dict, save_path)
        
    def save_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {'epoch': epoch, 'iter': current_iter, 'optimizers': [], 'schedulers': []}
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
                
            save_filename = f'{self.mode}_latest.state'
            save_path = os.path.join(self.model_path, save_filename)
            # print(f'[info] : save state in {save_path}')
            torch.save(state, save_path)
        
    
    def RUN(self):
        global current_iter
        global current_epoch
        current_iter, current_epoch = 0, 0
        print(f'INFO : Start training.')
        
        # setting torch run.
        torch.backends.cudnn.benchmark = True
        
        # load resume states if necessary
        if self.opt['path'].get('state'):
            device_id = torch.cuda.current_device()
            resume_state = torch.load(opt['path']['state'], map_location=lambda storage, loc: storage.cuda(device_id))
            current_epoch = resume_state['epoch']
            current_iter = resume_state['iter']
        else:
            resume_state = None
        
        total_epoch = self.opt['train'].get('total_epoch')
        while current_epoch < total_epoch:
            current_epoch += 1
            prog_bar = tqdm(enumerate(self.train_data_loader), total=len(self.train_data_loader))

            pixel_loss, perceptual_loss = 0, 0
            for step, (_data_dict) in prog_bar:
                current_iter += 1
                loss_dict = OrderedDict()
                
                self.set_data(_data_dict)
                
                # optimize net_g
                for p in self.net_D.parameters():
                    p.requires_grad = False

                self.optimizer_g.zero_grad()
                self.output, l_codebook, quant_stats = self.net_G(self.input)

                l_codebook = l_codebook * self.l_weight_codebook

                l_g_total = 0
                # pixel loss
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, self.input)
                    loss_dict['l_g_pix'] = l_g_pix.item()
                    pixel_loss += l_g_pix.item()
                    self.writer.add_scalar('pixel_loss', pixel_loss / (step+1), current_iter)
                    l_g_total += l_g_pix

                # perceptual loss
                if self.cri_perceptual:
                    l_g_percep = self.cri_perceptual(self.output, self.input)
                    loss_dict['l_g_percep'] = l_g_percep.item()
                    perceptual_loss += l_g_percep.item()
                    self.writer.add_scalar('perceptual_loss', perceptual_loss / (step+1), current_iter)
                    l_g_total += l_g_percep

                # gan loss
                if current_iter > self.net_d_start_iter:
                    # fake_g_pred = self.net_d(self.output_1024)
                    fake_g_pred = self.net_D(self.output)
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                    recon_loss = l_g_total
                    last_layer = self.net_G.generator.blocks[-1].weight
                    d_weight = self.calculate_adaptive_weight(recon_loss, l_g_gan, last_layer, disc_weight_max=1.0)
                    d_weight *= self.adopt_weight(1, current_iter, self.net_d_start_iter)
                    d_weight *= self.disc_weight # tamming setting 0.8
                    l_g_total += d_weight * l_g_gan
                    loss_dict['l_g_gan'] = d_weight * l_g_gan.item()

                l_g_total += l_codebook
                loss_dict['l_codebook'] = l_codebook.item()

                l_g_total.backward()
                self.optimizer_g.step()

                # optimize net_d
                if  current_iter > self.net_d_start_iter:
                    for p in self.net_D.parameters():
                        p.requires_grad = True

                    self.optimizer_d.zero_grad()
                    # real
                    real_d_pred = self.net_D(self.input)
                    l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                    loss_dict['l_d_real'] = l_d_real.item()
                    loss_dict['out_d_real'] = torch.mean(real_d_pred.detach()).item()
                    l_d_real.backward()
                    # fake
                    fake_d_pred = self.net_D(self.output.detach())
                    l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                    loss_dict['l_d_fake'] = l_d_fake.item()
                    loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach()).item()
                    l_d_fake.backward()
                    self.optimizer_d.step()
                
                if current_iter == 1 or current_iter % self.opt['inter_visualization'] == 0:
                    self.save_img()
                
                if current_iter % self.opt['inter_printlog'] == 0:
                    print(f'INFO : current_epoch {current_epoch}. current_iter {current_iter}')
                    print(f'INFO : loss {loss_dict}')
                    
                if current_iter % self.opt['inter_save_checkpoint'] == 0:
                    self.save_ckpt(self.net_G, 'net_g')
                    self.save_ckpt(self.net_D, 'net_d')
                    self.save_state(current_epoch, current_iter)
                    
                
  

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='super params to parse.')
    parser.add_argument('--mode', type=str, choices=['vqgan', 'enhance_teeth'])
    args = parser.parse_args()
    
    # load super params.
    print(f'Loaded params from {args.cfg}')
    opt = parse(args.cfg)
    
    # start train.
    pipeline = Train_pipeline(opt, mode=args.mode)
    pipeline.RUN()